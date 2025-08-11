from dataclasses import dataclass, field
@dataclass
class ScoringConfig:
    correct_score: float = 1.0
    incorrect_score: float = -1.0
    format_error_score: float = -1.0
    wo_bos_think: float = -2.0
    wo_eos_think: float = -0.5