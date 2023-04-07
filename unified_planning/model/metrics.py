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

import unified_planning as up
from fractions import Fraction
from typing import Dict, Optional, Union, Tuple


class PlanQualityMetric:
    """
    This is the base class of any metric for :class:`~unified_planning.model.Plan` quality.

    The addition of a `PlanQualityMetric` in a `Problem` restricts the set of valid `Plans` to only those who
    satisfy the semantic of the given metric, so a `Plan`, to be valid, not only needs to satisfy all the
    problem goals, but also the problem's quality metric.
    """

    pass


class MinimizeActionCosts(PlanQualityMetric):
    """
    This metric means that only the :class:`~unified_planning.model.Plan` minimizing the total cost of the :class:`Actions <unified_planning.model.Action>` is valid.

    The costs for each `Action` of the problem is stored in this quality metric.
    """

    def __init__(
        self,
        costs: Dict["up.model.Action", Optional["up.model.FNode"]],
        default: Optional["up.model.FNode"] = None,
    ):
        self.costs = costs
        self.default = default

    def __repr__(self):
        costs = {a.name: c for a, c in self.costs.items()}
        costs["default"] = self.default
        return f"minimize actions-cost: {costs}"

    def __eq__(self, other):
        return (
            isinstance(other, MinimizeActionCosts)
            and self.default == other.default
            and self.costs == other.costs
        )

    def __hash__(self):
        return hash(self.__class__.__name__)

    def get_action_cost(self, action: "up.model.Action") -> Optional["up.model.FNode"]:
        """
        Returns the cost of the given `Action`.

        :param action: The action of which cost must be retrieved.
        :return: The expression representing the cost of the given action. The retrieved cost might be `None`,
            meaning that `#TODO: add meaning of a None action cost`.
        """
        return self.costs.get(action, self.default)


class MinimizeSequentialPlanLength(PlanQualityMetric):
    """This metric means that the number of :func:`actions <unified_planning.plans.SequentialPlan.actions>` in the resulting :class:`~unified_planning.plans.SequentialPlan` must be minimized."""

    def __repr__(self):
        return "minimize sequential-plan-length"

    def __eq__(self, other):
        return isinstance(other, MinimizeSequentialPlanLength)

    def __hash__(self):
        return hash(self.__class__.__name__)


class MinimizeMakespan(PlanQualityMetric):
    """This metric means #TODO: explain what that metric means."""

    def __repr__(self):
        return "minimize makespan"

    def __eq__(self, other):
        return isinstance(other, MinimizeMakespan)

    def __hash__(self):
        return hash(self.__class__.__name__)


class MinimizeExpressionOnFinalState(PlanQualityMetric):
    """
    This metric means that the given expression must be minimized on the final state reached
    following the given :class:`~unified_planning.model.Plan`.
    """

    def __init__(self, expression: "up.model.FNode"):
        self.expression = expression

    def __repr__(self):
        return f"minimize {self.expression}"

    def __eq__(self, other):
        return (
            isinstance(other, MinimizeExpressionOnFinalState)
            and self.expression == other.expression
        )

    def __hash__(self):
        return hash(self.__class__.__name__)


class MaximizeExpressionOnFinalState(PlanQualityMetric):
    """
    This metric means that the given expression must be maximized on the final state reached
    following the given :class:`~unified_planning.model.Plan`.
    """

    def __init__(self, expression: "up.model.FNode"):
        self.expression = expression

    def __repr__(self):
        return f"maximize {self.expression}"

    def __eq__(self, other):
        return (
            isinstance(other, MaximizeExpressionOnFinalState)
            and self.expression == other.expression
        )

    def __hash__(self):
        return hash(self.__class__.__name__)


class Oversubscription(PlanQualityMetric):
    """
    This metric means that only the plans maximizing the total gain of the achieved `goals` is valid.

    The gained value for each fulfilled `goal` of the problem is stored in this quality metric.
    """

    def __init__(self, goals: Dict["up.model.FNode", Union[Fraction, int]]):
        goals = dict(
            (k, f.numerator if f.denominator == 1 else f) for (k, f) in goals.items()
        )
        self.goals = goals

    def __repr__(self):
        return f"oversubscription planning goals: {self.goals}"

    def __eq__(self, other):
        return isinstance(other, Oversubscription) and self.goals == other.goals

    def __hash__(self):
        return hash(self.__class__.__name__)


class TemporalOversubscription(PlanQualityMetric):
    """
    This metric means that only the plans maximizing the total gain of the achieved `goals` is valid.

    The gained value for each fulfilled `goal` of the problem is stored in this quality metric.
    """

    def __init__(
        self,
        goals: Dict[
            Tuple["up.model.timing.TimeInterval", "up.model.FNode"],
            Union[Fraction, int],
        ],
    ):
        goals = dict(
            (k, f.numerator if f.denominator == 1 else f) for (k, f) in goals.items()
        )
        self.goals = goals

    def __repr__(self):
        return f"oversubscription planning timed goals: {self.goals}"

    def __eq__(self, other):
        return isinstance(other, TemporalOversubscription) and self.goals == other.goals

    def __hash__(self):
        return hash(self.__class__.__name__)
