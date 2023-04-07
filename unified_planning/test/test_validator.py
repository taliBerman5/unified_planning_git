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

import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.model.problem_kind import (
    classical_kind,
    general_numeric_kind,
    bounded_types_kind,
)
from unified_planning.test import TestCase, main, skipIfNoPlanValidatorForProblemKind
from unified_planning.test.examples import get_example_problems


class TestPlanValidator(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        self.problems = get_example_problems()

    @skipIfNoPlanValidatorForProblemKind(classical_kind)
    def test_basic(self):
        problem, plan = self.problems["basic"]

        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            self.assertNotEqual(validator, None)

            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.VALID)

            plan = up.plans.SequentialPlan([])
            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.INVALID)

    @skipIfNoPlanValidatorForProblemKind(
        classical_kind.union(general_numeric_kind).union(bounded_types_kind)
    )
    def test_robot(self):
        problem, plan = self.problems["robot"]

        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            self.assertNotEqual(validator, None)

            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.VALID)

            plan = up.plans.SequentialPlan([])
            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.INVALID)

    @skipIfNoPlanValidatorForProblemKind(classical_kind)
    def test_robot_loader(self):
        problem, plan = self.problems["robot_loader"]

        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            self.assertNotEqual(validator, None)

            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.VALID)

            plan = up.plans.SequentialPlan([])
            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.INVALID)

    @skipIfNoPlanValidatorForProblemKind(classical_kind)
    def test_robot_loader_adv(self):
        problem, plan = self.problems["robot_loader_adv"]

        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            self.assertNotEqual(validator, None)

            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.VALID)

            plan = up.plans.SequentialPlan([])
            res = validator.validate(problem, plan)
            self.assertEqual(res.status, up.engines.ValidationResultStatus.INVALID)


if __name__ == "__main__":
    main()
