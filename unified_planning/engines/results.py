# Copyright 2021 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""This module defines the PlanGenerationResult class."""


import unified_planning as up
from unified_planning.exceptions import UPUsageError
from unified_planning.model import AbstractProblem
from unified_planning.plans import ActionInstance
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Optional, List


class ValidationResultStatus(Enum):
    """
    Enum representing the 2 possible values in the `status` field of a :class:`~unified_planning.engines.ValidationResult`:
    VALID or INVALID.
    """

    VALID = (
        auto()
    )  # The plan is valid for the problem, it satisfies all the hard constraints
    INVALID = (
        auto()
    )  # The plan is invalid for the problem, it does not satisfy all the hard constraints

    def __bool__(self):
        if self == ValidationResultStatus.VALID:
            return True
        else:
            return False


class PlanGenerationResultStatus(Enum):
    """
    Enum representing the 9 possible values in the status field of a :class:`~unified_planning.engines.PlanGenerationResult`:
    SOLVED_SATISFICING        -> Valid plan found.
    SOLVED_OPTIMALLY          -> Optimal plan found.
    UNSOLVABLE_PROVEN         -> The problem is impossible, no valid plan exists.
    UNSOLVABLE_INCOMPLETELY   -> The planner could not find a plan, but it's not sure that
                                 the problem is impossible (The planner is incomplete)
    TIMEOUT                   -> The planner ran out of time
    MEMOUT                    -> The planner ran out of memory
    INTERNAL_ERROR            -> The planner had an internal error
    UNSUPPORTED_PROBLEM       -> The problem given is not supported by the planner
    INTERMEDIATE              -> The report is not a final one but it's given through the callback function
    """

    SOLVED_SATISFICING = auto()  # Valid plan found.
    SOLVED_OPTIMALLY = auto()  # Optimal plan found.
    UNSOLVABLE_PROVEN = auto()  # The problem is impossible, no valid plan exists.
    UNSOLVABLE_INCOMPLETELY = (
        auto()
    )  # The planner could not find a plan, but it's not sure that the problem is impossible (The planner is incomplete)
    TIMEOUT = auto()  # The planner ran out of time
    MEMOUT = auto()  # The planner ran out of memory
    INTERNAL_ERROR = auto()  # The planner had an internal error
    UNSUPPORTED_PROBLEM = auto()  # The problem given is not supported by the planner
    INTERMEDIATE = (
        auto()
    )  # The report is not a final one but it's an intermediate anytime result


POSITIVE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    ]
)

NEGATIVE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
        PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
        PlanGenerationResultStatus.UNSUPPORTED_PROBLEM,
    ]
)


class LogLevel(Enum):
    """
    Enum representing the 4 possible values in the verbosity level of a :class:`~unified_planning.engines.LogMessage`:
    DEBUG, INFO, WARNING and ERROR
    """

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class LogMessage:
    """
    This class is composed by a message and the Enum LogLevel indicating
    this message level, like Debug, Info, Warning or Error.
    """

    level: LogLevel
    message: str

    def __str__(self):
        return f"[{self.level.name}] {self.message}"


@dataclass
class Result:
    """This class represents the base class for results given by the engines to the user."""

    def is_definitive_result(self, *args) -> bool:
        """This predicate should state if the Result is definitive or if it can be improved."""
        raise NotImplementedError


@dataclass
class PlanGenerationResult(Result):
    """Class that represents the result of a plan generation call."""

    status: PlanGenerationResultStatus
    plan: Optional["up.plans.Plan"]
    engine_name: str
    metrics: Optional[Dict[str, str]] = field(default=None)
    log_messages: Optional[List[LogMessage]] = field(default=None)

    def __post__init(self):
        # Checks that plan and status are consistent
        if self.status in POSITIVE_OUTCOMES and self.plan is None:
            raise UPUsageError(
                f"The Result status is {str(self.status)} but no plan is set."
            )
        elif self.status in NEGATIVE_OUTCOMES and self.plan is not None:
            raise UPUsageError(
                f"The Result status is {str(self.status)} but the plan is {str(self.plan)}.\nWith this status the plan must be None."
            )
        return self

    def is_definitive_result(self, *args) -> bool:
        optimality_required = False
        if len(args) > 0:
            optimality_required = (
                len(args[0].quality_metrics) > 0
            )  # Require optimality if the problem has at least one quality metric.
        return (
            self.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY
            or self.status == PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            or (
                optimality_required
                and self.status == PlanGenerationResultStatus.SOLVED_SATISFICING
            )
        )


@dataclass
class ValidationResult(Result):
    """Class that represents the result of a validate call."""

    status: ValidationResultStatus
    engine_name: str
    log_messages: Optional[List[LogMessage]] = field(default=None)

    def is_definitive_result(self, *args) -> bool:
        return True

    def __bool__(self):
        if self.status == ValidationResultStatus.VALID:
            return True
        else:
            return False


@dataclass
class CompilerResult(Result):
    """Class that represents the result of a compile call."""

    problem: Optional[AbstractProblem]
    map_back_action_instance: Optional[
        Callable[[ActionInstance], Optional[ActionInstance]]
    ]
    engine_name: str
    log_messages: Optional[List[LogMessage]] = field(default=None)

    def _post_init(self):
        # Check that compiled problem and map_back_action_instance are consistent with each other
        if self.problem is None and self.map_back_action_instance is not None:
            raise UPUsageError(
                f"The compiled Problem is None but the map_back_action_instance Callable is not None."
            )
        if self.problem is not None and self.map_back_action_instance is None:
            raise UPUsageError(
                f"The compiled Problem is {str(self.problem)} but the map_back_action_instance Callable is None."
            )

    def is_definitive_result(self, *args) -> bool:
        return self.problem is not None
