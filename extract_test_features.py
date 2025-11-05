#!/usr/bin/env python3
"""Extract assembly-derived features for Ibex regression tests.

This utility consumes ``coverage_labels.jsonl`` produced by
``coverage_labeler.py`` and inspects each test's generated RISC-V
assembly listing (``test.S``). The resulting feature vectors combine
static code characteristics with the coverage labels, making it easier to
experiment with ranking and ML pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Optional, Sequence, Set

# Opcodes grouped by functional category. The lists are intentionally
# conservative; any opcode that is not recognised falls back to the
# "other" bucket but still contributes to aggregate ratios.
BRANCH_OPS: Set[str] = {
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "beqz",
    "bnez",
}

JUMP_OPS: Set[str] = {
    "jal",
    "jalr",
    "j",
    "jr",
    "ret",
    "call",
    "tail",
}

LOAD_OPS: Set[str] = {
    "lb",
    "lbu",
    "lh",
    "lhu",
    "lw",
    "lwu",
    "ld",
    "flw",
    "fld",
}

STORE_OPS: Set[str] = {
    "sb",
    "sh",
    "sw",
    "sd",
    "fsw",
    "fsd",
}

CSR_OPS: Set[str] = {
    "csrrw",
    "csrrs",
    "csrrc",
    "csrrwi",
    "csrrsi",
    "csrrci",
    "csrw",
    "csrs",
    "csrc",
    "csrwi",
    "csrsi",
    "csrci",
}

PRIVILEGED_OPS: Set[str] = {
    "ecall",
    "ebreak",
    "mret",
    "sret",
    "uret",
    "dret",
    "wfi",
    "sfence.vma",
    "sfence.vmda",
    "hfence.gvma",
    "hfence.vvma",
    "fence",
    "fence.i",
}

ATOMIC_OPS: Set[str] = {
    "lr.w",
    "lr.d",
    "sc.w",
    "sc.d",
    "amoadd.w",
    "amoadd.d",
    "amoxor.w",
    "amoxor.d",
    "amoand.w",
    "amoand.d",
    "amoor.w",
    "amoor.d",
    "amomax.w",
    "amomax.d",
    "amomaxu.w",
    "amomaxu.d",
    "amomin.w",
    "amomin.d",
    "amominu.w",
    "amominu.d",
    "amoswap.w",
    "amoswap.d",
}

ALU_OPS: Set[str] = {
    "add",
    "addi",
    "sub",
    "and",
    "andi",
    "or",
    "ori",
    "xor",
    "xori",
    "sll",
    "slli",
    "srl",
    "srli",
    "sra",
    "srai",
    "slti",
    "sltiu",
    "slt",
    "sltu",
    "lui",
    "auipc",
    "not",
    "neg",
    "negw",
    "sext.w",
    "zext.b",
    "zext.h",
    "rol",
    "ror",
    "orn",
    "xnor",
    "min",
    "minu",
    "max",
    "maxu",
    "pack",
    "packh",
    "packu",
    "addw",
    "addiw",
    "sllw",
    "srlw",
    "sraw",
}

MULDIV_OPS: Set[str] = {
    "mul",
    "mulh",
    "mulhsu",
    "mulhu",
    "mulw",
    "div",
    "divu",
    "divw",
    "divuw",
    "rem",
    "remu",
    "remw",
    "remuw",
}

PSEUDO_MOVE_OPS: Set[str] = {
    "li",
    "la",
    "lla",
    "mv",
    "nop",
}

FLOATING_OPS: Set[str] = {
    "fadd.s",
    "fsub.s",
    "fmul.s",
    "fdiv.s",
    "fsqrt.s",
    "fsgnj.s",
    "fsgnjn.s",
    "fsgnjx.s",
    "fmin.s",
    "fmax.s",
    "fcvt.w.s",
    "fcvt.wu.s",
    "fcvt.s.w",
    "fcvt.s.wu",
    "fmv.w.x",
    "fmv.x.w",
    "fclass.s",
    "fsgnj",
    "fsgnjn",
    "fsgnjx",
    "flw",
    "fsw",
}

# Ratios and counters will be projected for this ordered list of keys.
CATEGORY_KEYS: Sequence[str] = (
    "branch",
    "jump",
    "load",
    "store",
    "csr",
    "privileged",
    "atomic",
    "alu",
    "muldiv",
    "pseudo_move",
    "floating",
)

class AssemblyFeatureExtractor:
    """Incrementally accumulate features from RISC-V assembly lines."""

    def __init__(self) -> None:
        self.instruction_count = 0
        self.unique_opcodes: Set[str] = set()
        self.category_counts: Counter[str] = Counter()
        self.csr_targets: Set[str] = set()
        self.immediates: List[int] = []
        self.gpr_indices: Set[int] = set()
        self.fpr_indices: Set[int] = set()
        self.compressed_count = 0

    def _categorise(self, opcode: str) -> Set[str]:
        categories: Set[str] = set()
        base = opcode.lower()
        if base.startswith("c."):
            categories.add("compressed")
            base = base[2:]
        if base in BRANCH_OPS:
            categories.add("branch")
        if base in JUMP_OPS:
            categories.add("jump")
        if base in LOAD_OPS:
            categories.add("load")
        if base in STORE_OPS:
            categories.add("store")
        if base in CSR_OPS:
            categories.add("csr")
        if base in PRIVILEGED_OPS:
            categories.add("privileged")
        if base in ATOMIC_OPS:
            categories.add("atomic")
        if base in FLOATING_OPS:
            categories.add("floating")
        if base in MULDIV_OPS:
            categories.add("muldiv")
        if base in ALU_OPS:
            categories.add("alu")
        if base in PSEUDO_MOVE_OPS:
            categories.add("pseudo_move")
        return categories

    @staticmethod
    def _tokenise(line: str) -> List[str]:
        clean = line
        for marker in ("#", "//"):
            if marker in clean:
                clean = clean.split(marker, 1)[0]
        clean = clean.strip()
        if not clean:
            return []
        if clean.startswith("."):
            return []
        if ":" in clean:
            label_split = clean.split(":", 1)
            tail = label_split[1].strip()
            if not tail:
                return []
            clean = tail
        clean = clean.replace(",", " ")
        clean = clean.replace("(", " ")
        clean = clean.replace(")", " ")
        tokens = [tok for tok in clean.split() if tok]
        return tokens

    @staticmethod
    def _parse_immediate(token: str) -> Optional[int]:
        tok = token.lower()
        if tok.startswith("0x") or tok.startswith("-0x"):
            try:
                return int(tok, 16)
            except ValueError:
                return None
        try:
            if tok.startswith("0b") or tok.startswith("-0b"):
                return int(tok, 2)
            return int(tok, 10)
        except ValueError:
            return None

    def process_line(self, line: str) -> None:
        tokens = self._tokenise(line)
        if not tokens:
            return
        opcode = tokens[0].lower()
        self.instruction_count += 1
        self.unique_opcodes.add(opcode)

        categories = self._categorise(opcode)
        for category in categories:
            if category == "compressed":
                self.compressed_count += 1
            else:
                self.category_counts[category] += 1

        operands = tokens[1:]

        if "csr" in categories and operands:
            self.csr_targets.add(operands[0].lower())

        for operand in operands:
            operand_lower = operand.lower()
            if operand_lower.startswith("x") and operand_lower[1:].isdigit():
                self.gpr_indices.add(int(operand_lower[1:]))
            elif operand_lower.startswith("f") and operand_lower[1:].isdigit():
                self.fpr_indices.add(int(operand_lower[1:]))
            value = self._parse_immediate(operand_lower)
            if value is not None:
                self.immediates.append(value)

    def process_file(self, asm_path: Path) -> None:
        with asm_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                self.process_line(line)

    def as_features(self) -> Dict[str, object]:
        features: Dict[str, object] = {}
        features["instruction_count"] = self.instruction_count
        features["unique_opcode_count"] = len(self.unique_opcodes)
        for key in CATEGORY_KEYS:
            features[f"{key}_count"] = self.category_counts.get(key, 0)
        features["compressed_count"] = self.compressed_count
        features["csr_unique_targets"] = len(self.csr_targets)
        features["unique_gprs"] = len(self.gpr_indices)
        features["unique_fprs"] = len(self.fpr_indices)

        total = max(self.instruction_count, 1)
        for key in CATEGORY_KEYS:
            features[f"{key}_fraction"] = self.category_counts.get(key, 0) / total
        features["compressed_fraction"] = self.compressed_count / total

        if self.immediates:
            abs_values = [abs(val) for val in self.immediates]
            features["immediate_count"] = len(self.immediates)
            features["immediate_max_abs"] = max(abs_values)
            features["immediate_mean_abs"] = fmean(abs_values)
        else:
            features["immediate_count"] = 0
            features["immediate_max_abs"] = 0
            features["immediate_mean_abs"] = 0.0

        features["max_gpr_index"] = max(self.gpr_indices) if self.gpr_indices else -1
        features["max_fpr_index"] = max(self.fpr_indices) if self.fpr_indices else -1
        return features


def load_labels(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def discover_test_root(coverage_path: Path) -> Path:
    return coverage_path.parent.parent


def build_feature_row(entry: Dict[str, object], extractor: AssemblyFeatureExtractor) -> Dict[str, object]:
    base = {
        "testdotseed": entry.get("testdotseed"),
        "coverage_path": entry.get("coverage_path"),
        "label": entry.get("label", 0),
    }
    features = extractor.as_features()
    covered_deltas = entry.get("covered_deltas", {})
    for metric, value in covered_deltas.items():
        base[f"delta_{metric}"] = value
    base["covergroup_delta"] = entry.get("covergroup_delta", 0.0)
    base["total_delta"] = sum(float(v) for v in covered_deltas.values())
    base.update(features)
    return base


def write_jsonl(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.touch()
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RISC-V assembly features for Ibex regression tests.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("coverage_labels.jsonl"),
        help="Path to coverage_labels.jsonl produced by coverage_labeler.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coverage_features.jsonl"),
        help="Output file for feature vectors (JSONL or CSV).",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format; defaults to JSONL for streaming workflows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of records processed (debug aid).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a test assembly file is missing instead of skipping it.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.labels.exists():
        raise SystemExit(f"Labels file {args.labels} does not exist.")

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []

    for idx, entry in enumerate(load_labels(args.labels), start=1):
        coverage_path_str = entry.get("coverage_path")
        if not coverage_path_str:
            skipped.append(f"record {idx}: missing coverage_path")
            if args.strict:
                raise SystemExit(f"Missing coverage_path for record {idx}.")
            continue
        coverage_path = Path(str(coverage_path_str)).resolve()
        asm_path = discover_test_root(coverage_path) / "test.S"
        if not asm_path.exists():
            skipped.append(str(asm_path))
            if args.strict:
                raise SystemExit(f"Assembly file not found: {asm_path}")
            continue
        extractor = AssemblyFeatureExtractor()
        extractor.process_file(asm_path)
        rows.append(build_feature_row(entry, extractor))
        if args.limit and len(rows) >= args.limit:
            break

    if args.format == "jsonl":
        write_jsonl(rows, args.output)
    else:
        write_csv(rows, args.output)

    if skipped:
        print(
            f"Skipped {len(skipped)} test(s) without assembly listings. "
            "Use --strict to treat this as an error.",
            file=sys.stderr,
        )
    print(f"Wrote {len(rows)} feature rows to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
