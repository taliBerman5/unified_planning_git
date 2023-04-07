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


import unified_planning.model.walkers as walkers

import unified_planning.environment
from unified_planning.model.walkers.identitydag import IdentityDagWalker
from unified_planning.model.fnode import FNode
from unified_planning.model.operators import OperatorKind
from unified_planning.model.expression import Expression
from unified_planning.exceptions import UPTypeError
from typing import List, Dict


class Substituter(IdentityDagWalker):
    """Performs substitution into an expression"""

    def __init__(self, environment: "unified_planning.environment.Environment"):
        IdentityDagWalker.__init__(self, environment, True)
        self.environment = environment
        self.manager = environment.expression_manager
        self.type_checker = environment.type_checker

    def _get_key(self, expression, **kwargs):
        return expression

    def substitute(
        self, expression: FNode, substitutions: Dict[Expression, Expression] = {}
    ) -> FNode:
        """
        Performs substitution into the given expression.

        The substitutions are made top-down in the expression tree and the substitution is not
        applied to the substituted expressions.

        :param expression: The target expression for the substitution.
        :param substitutions: The map containing the substitutions, every time a key is found,
            it is substituted with it's value.
        :return: The expression where every key expression is substituted with it's value.

        Lets consider the examples:
        f = a & b
        subs = {a -> c, (c & b) -> d, (a & b) -> c}
        substitute(f, subs) = c

        f = a
        subs = {a -> c, c -> d}
        substitute(f, subs) = c

        f = a & b
        subs = {a -> 5, b -> c}
        substitute(f, subs) raises an UPTypeError

        Note that, since subs is a dictionary:
        f = a
        subs = {a -> b, a -> c}
        substitute(f, subs) = c
        """

        if len(substitutions) == 0:
            return expression
        new_substitutions: Dict[FNode, FNode] = {}
        for k, v in substitutions.items():
            new_k, new_v = self.manager.auto_promote(k, v)
            if new_k.type.is_compatible(new_v.type):
                new_substitutions[new_k] = new_v
            else:
                raise UPTypeError(
                    f"The expression type of {str(k)} is not compatible with the given substitution {str(v)}"
                )
        return self.walk(expression, subs=new_substitutions)

    @walkers.handles(OperatorKind)
    def walk_replace_or_identity(
        self,
        expression: FNode,
        args: List[FNode],
        subs: Dict[FNode, FNode] = {},
        **kwargs,
    ) -> FNode:
        res = subs.get(expression, None)
        if res is not None:
            return res
        else:
            return IdentityDagWalker.super(self, expression, args, **kwargs)
