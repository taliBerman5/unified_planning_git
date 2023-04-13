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
"""
This module defines the `Action` base class and some of his extensions.
An `Action` has a `name`, a `list` of `Parameter`, a `list` of `preconditions`
and a `list` of `effects`.
"""


import unified_planning as up
from unified_planning.environment import get_environment, Environment
import unified_planning.shortcuts
from unified_planning.exceptions import (
    UPTypeError,
    UPUnboundedVariablesError,
    UPProblemDefinitionError,
    UPConflictingEffectsException,
    UPUsageError,
)
from fractions import Fraction
from typing import Dict, List, Set, Tuple, Union, Optional, cast, Callable
from collections import OrderedDict

from unified_planning.model.mixins.timed_conds_effs import TimedCondsEffs


class Action:
    """This is the `Action` interface."""

    def __init__(
        self,
        _name: str,
        _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
        _env: Optional[Environment] = None,
        **kwargs: "up.model.types.Type",
    ):
        self._environment = get_environment(_env)
        self._name = _name
        self._parameters: "OrderedDict[str, up.model.parameter.Parameter]" = (
            OrderedDict()
        )
        if _parameters is not None:
            assert len(kwargs) == 0
            for n, t in _parameters.items():
                assert self._environment.type_manager.has_type(
                    t
                ), "type of parameter does not belong to the same environment of the action"
                self._parameters[n] = up.model.parameter.Parameter(
                    n, t, self._environment
                )
        else:
            for n, t in kwargs.items():
                assert self._environment.type_manager.has_type(
                    t
                ), "type of parameter does not belong to the same environment of the action"
                self._parameters[n] = up.model.parameter.Parameter(
                    n, t, self._environment
                )

    def __eq__(self, oth: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    @property
    def environment(self) -> Environment:
        """Returns this `Action` `Environment`."""
        return self._environment

    @property
    def name(self) -> str:
        """Returns the `Action` `name`."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        """Sets the `Action` `name`."""
        self._name = new_name

    @property
    def parameters(self) -> List["up.model.parameter.Parameter"]:
        """Returns the `list` of the `Action parameters`."""
        return list(self._parameters.values())

    def parameter(self, name: str) -> "up.model.parameter.Parameter":
        """
        Returns the `parameter` of the `Action` with the given `name`.

        Example
        -------
        >>> from unified_planning.shortcuts import *
        >>> location_type = UserType("Location")
        >>> move = InstantaneousAction("move", source=location_type, target=location_type)
        >>> move.parameter("source")  # return the "source" parameter of the action, with type "Location"
        Location source
        >>> move.parameter("target")
        Location target

        If a parameter's name (1) does not conflict with an existing attribute of `Action` and (2) does not start with '_'
        it can also be accessed as if it was an attribute of the action. For instance:

        >>> move.source
        Location source

        :param name: The `name` of the target `parameter`.
        :return: The `parameter` of the `Action` with the given `name`.
        """
        if name not in self._parameters:
            raise ValueError(f"Action '{self.name}' has no parameter '{name}'")
        return self._parameters[name]

    def __getattr__(self, parameter_name: str) -> "up.model.parameter.Parameter":
        if parameter_name.startswith("_"):
            # guard access as pickling relies on attribute error to be thrown even when
            # no attributes of the object have been set.
            # In this case accessing `self._name` or `self._parameters`, would re-invoke __getattr__
            raise AttributeError(f"Action has no attribute '{parameter_name}'")
        if parameter_name not in self._parameters:
            raise AttributeError(
                f"Action '{self.name}' has no attribute or parameter '{parameter_name}'"
            )
        return self._parameters[parameter_name]

    def is_conditional(self) -> bool:
        """Returns `True` if the `Action` has `conditional effects`, `False` otherwise."""
        raise NotImplementedError


class InstantaneousAction(Action):
    """Represents an instantaneous action."""

    def __init__(
        self,
        _name: str,
        _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
        _env: Optional[Environment] = None,
        **kwargs: "up.model.types.Type",
    ):
        Action.__init__(self, _name, _parameters, _env, **kwargs)
        self._preconditions: List["up.model.fnode.FNode"] = []
        self._effects: List[up.model.effect.Effect] = []
        self._simulated_effect: Optional[up.model.effect.SimulatedEffect] = None
        # fluent assigned is the mapping of the fluent to it's value if it is an unconditional assignment
        self._fluents_assigned: Dict[
            "up.model.fnode.FNode", "up.model.fnode.FNode"
        ] = {}
        # fluent_inc_dec is the set of the fluents that have an unconditional increase or decrease
        self._fluents_inc_dec: Set["up.model.fnode.FNode"] = set()
    def __repr__(self) -> str:
        s = []
        s.append(f"action {self.name}")
        first = True
        for p in self.parameters:
            if first:
                s.append("(")
                first = False
            else:
                s.append(", ")
            s.append(str(p))
        if not first:
            s.append(")")
        s.append(" {\n")
        s.append("    preconditions = [\n")
        for c in self.preconditions:
            s.append(f"      {str(c)}\n")
        s.append("    ]\n")
        s.append("    effects = [\n")
        for e in self.effects:
            s.append(f"      {str(e)}\n")
        s.append("    ]\n")
        if self._simulated_effect is not None:
            s.append(f"    simulated effect = {self._simulated_effect}\n")
        s.append("  }")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, InstantaneousAction):
            cond = (
                self._environment == oth._environment
                and self._name == oth._name
                and self._parameters == oth._parameters
            )
            return (
                cond
                and set(self._preconditions) == set(oth._preconditions)
                and set(self._effects) == set(oth._effects)
                and self._simulated_effect == oth._simulated_effect
            )
        else:
            return False

    def __hash__(self) -> int:
        res = hash(self._name)
        for ap in self._parameters.items():
            res += hash(ap)
        for p in self._preconditions:
            res += hash(p)
        for e in self._effects:
            res += hash(e)
        res += hash(self._simulated_effect)
        return res

    def clone(self):
        new_params = OrderedDict(
            (param_name, param.type) for param_name, param in self._parameters.items()
        )
        new_instantaneous_action = InstantaneousAction(
            self._name, new_params, self._environment
        )
        new_instantaneous_action._preconditions = self._preconditions[:]
        new_instantaneous_action._effects = [e.clone() for e in self._effects]
        new_instantaneous_action._fluents_assigned = self._fluents_assigned.copy()
        new_instantaneous_action._fluents_inc_dec = self._fluents_inc_dec.copy()
        new_instantaneous_action._simulated_effect = self._simulated_effect
        return new_instantaneous_action

    @property
    def preconditions(self) -> List["up.model.fnode.FNode"]:
        """Returns the `list` of the `Action` `preconditions`."""
        return self._preconditions

    def clear_preconditions(self):
        """Removes all the `Action preconditions`"""
        self._preconditions = []

    @property
    def effects(self) -> List["up.model.effect.Effect"]:
        """Returns the `list` of the `Action effects`."""
        return self._effects

    def clear_effects(self):
        """Removes all the `Action's effects`."""
        self._effects = []
        self._fluents_assigned = {}
        self._fluents_inc_dec = set()
        self._simulated_effect = None

    @property
    def conditional_effects(self) -> List["up.model.effect.Effect"]:
        """Returns the `list` of the `action conditional effects`.

        IMPORTANT NOTE: this property does some computation, so it should be called as
        seldom as possible."""
        return [e for e in self._effects if e.is_conditional()]

    def is_conditional(self) -> bool:
        """Returns `True` if the `action` has `conditional effects`, `False` otherwise."""
        return any(e.is_conditional() for e in self._effects)

    @property
    def unconditional_effects(self) -> List["up.model.effect.Effect"]:
        """Returns the `list` of the `action unconditional effects`.

        IMPORTANT NOTE: this property does some computation, so it should be called as
        seldom as possible."""
        return [e for e in self._effects if not e.is_conditional()]

    def add_precondition(
        self,
        precondition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool,
        ],
    ):
        """
        Adds the given expression to `action's preconditions`.

        :param precondition: The expression that must be added to the `action's preconditions`.
        """
        (precondition_exp,) = self._environment.expression_manager.auto_promote(
            precondition
        )
        assert self._environment.type_checker.get_type(precondition_exp).is_bool_type()
        if precondition_exp == self._environment.expression_manager.TRUE():
            return
        free_vars = self._environment.free_vars_oracle.get_free_variables(
            precondition_exp
        )
        if len(free_vars) != 0:
            raise UPUnboundedVariablesError(
                f"The precondition {str(precondition_exp)} has unbounded variables:\n{str(free_vars)}"
            )
        if precondition_exp not in self._preconditions:
            self._preconditions.append(precondition_exp)

    def add_effect(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
    ):
        """
        Adds the given `assignment` to the `action's effects`.

        :param fluent: The `fluent` of which `value` is modified by the `assignment`.
        :param value: The `value` to assign to the given `fluent`.
        :param condition: The `condition` in which this `effect` is applied; the default
            value is `True`.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_effect must be a Fluent or a FluentExp"
            )
        if not self._environment.type_checker.get_type(condition_exp).is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            # Value is not assignable to fluent (its type is not a subset of the fluent's type).
            raise UPTypeError(
                f"InstantaneousAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        self._add_effect_instance(
            up.model.effect.Effect(fluent_exp, value_exp, condition_exp)
        )

    def add_increase_effect(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
    ):
        """
        Adds the given `increase effect` to the `action's effects`.

        :param fluent: The `fluent` which `value` is increased.
        :param value: The given `fluent` is incremented by the given `value`.
        :param condition: The `condition` in which this `effect` is applied; the default
            value is `True`.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_increase_effect must be a Fluent or a FluentExp"
            )
        if not condition_exp.type.is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError(
                f"InstantaneousAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        if not fluent_exp.type.is_int_type() and not fluent_exp.type.is_real_type():
            raise UPTypeError("Increase effects can be created only on numeric types!")
        self._add_effect_instance(
            up.model.effect.Effect(
                fluent_exp,
                value_exp,
                condition_exp,
                kind=up.model.effect.EffectKind.INCREASE,
            )
        )

    def add_decrease_effect(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
    ):
        """
        Adds the given `decrease effect` to the `action's effects`.

        :param fluent: The `fluent` which value is decreased.
        :param value: The given `fluent` is decremented by the given `value`.
        :param condition: The `condition` in which this `effect` is applied; the default
            value is `True`.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_decrease_effect must be a Fluent or a FluentExp"
            )
        if not condition_exp.type.is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError(
                f"InstantaneousAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        if not fluent_exp.type.is_int_type() and not fluent_exp.type.is_real_type():
            raise UPTypeError("Decrease effects can be created only on numeric types!")
        self._add_effect_instance(
            up.model.effect.Effect(
                fluent_exp,
                value_exp,
                condition_exp,
                kind=up.model.effect.EffectKind.DECREASE,
            )
        )

    def _add_effect_instance(self, effect: "up.model.effect.Effect"):
        assert (
            effect.environment == self._environment
        ), "effect does not have the same environment of the action"
        up.model.effect.check_conflicting_effects(
            effect,
            None,
            self._simulated_effect,
            self._fluents_assigned,
            self._fluents_inc_dec,
            "action",
        )
        self._effects.append(effect)

    @property
    def simulated_effect(self) -> Optional["up.model.effect.SimulatedEffect"]:
        """Returns the `action` `simulated effect`."""
        return self._simulated_effect

    def set_simulated_effect(self, simulated_effect: "up.model.effect.SimulatedEffect"):
        """
        Sets the given `simulated effect` as the only `action's simulated effect`.

        :param simulated_effect: The `SimulatedEffect` instance that must be set as this `action`'s only
            `simulated effect`.
        """
        up.model.effect.check_conflicting_simulated_effects(
            simulated_effect,
            None,
            self._fluents_assigned,
            self._fluents_inc_dec,
            "action",
        )
        self._simulated_effect = simulated_effect

    def _set_preconditions(self, preconditions: List["up.model.fnode.FNode"]):
        self._preconditions = preconditions


class DurativeAction(Action, TimedCondsEffs):
    """Represents a durative action."""

    def __init__(
        self,
        _name: str,
        _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
        _env: Optional[Environment] = None,
        **kwargs: "up.model.types.Type",
    ):
        Action.__init__(self, _name, _parameters, _env, **kwargs)
        TimedCondsEffs.__init__(self, _env)
        self._duration: "up.model.timing.DurationInterval" = (
            up.model.timing.FixedDuration(self._environment.expression_manager.Int(0))
        )

    def __repr__(self) -> str:
        s = []
        s.append(f"durative action {self.name}")
        first = True
        for p in self.parameters:
            if first:
                s.append("(")
                first = False
            else:
                s.append(", ")
            s.append(str(p))
        if not first:
            s.append(")")
        s.append(" {\n")
        s.append(f"    duration = {str(self._duration)}\n")
        s.append("    conditions = [\n")
        for i, cl in self.conditions.items():
            s.append(f"      {str(i)}:\n")
            for c in cl:
                s.append(f"        {str(c)}\n")
        s.append("    ]\n")
        s.append("    effects = [\n")
        for t, el in self.effects.items():
            s.append(f"      {str(t)}:\n")
            for e in el:
                s.append(f"        {str(e)}:\n")
        s.append("    ]\n")
        s.append("    simulated effects = [\n")
        for t, se in self.simulated_effects.items():
            s.append(f"      {str(t)}: {se}\n")
        s.append("    ]\n")
        s.append("  }")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if not isinstance(oth, DurativeAction):
            return False
        if (
            self._environment != oth._environment
            or self._name != oth._name
            or self._parameters != oth._parameters
            or self._duration != oth._duration
        ):
            return False
        if not TimedCondsEffs.__eq__(self, oth):
            return False
        return True

    def __hash__(self) -> int:
        res = hash(self._name) + hash(self._duration)
        for ap in self._parameters.items():
            res += hash(ap)
        res += TimedCondsEffs.__hash__(self)
        return res

    def clone(self):
        new_params = OrderedDict(
            (param_name, param.type) for param_name, param in self._parameters.items()
        )
        new_durative_action = DurativeAction(self._name, new_params, self._environment)
        new_durative_action._duration = self._duration

        TimedCondsEffs._clone_to(self, new_durative_action)
        return new_durative_action

    @property
    def duration(self) -> "up.model.timing.DurationInterval":
        """Returns the `action` `duration interval`."""
        return self._duration

    def set_duration_constraint(self, duration: "up.model.timing.DurationInterval"):
        """
        Sets the `duration interval` for this `action`.

        :param duration: The new `duration interval` of this `action`.
        """
        lower, upper = duration.lower, duration.upper
        tlower = self._environment.type_checker.get_type(lower)
        tupper = self._environment.type_checker.get_type(upper)
        assert tlower.is_int_type() or tlower.is_real_type()
        assert tupper.is_int_type() or tupper.is_real_type()
        if (
            lower.is_constant()
            and upper.is_constant()
            and (
                upper.constant_value() < lower.constant_value()
                or (
                    upper.constant_value() == lower.constant_value()
                    and (duration.is_left_open() or duration.is_right_open())
                )
            )
        ):
            raise UPProblemDefinitionError(
                f"{duration} is an empty interval duration of action: {self.name}."
            )
        self._duration = duration

    def set_fixed_duration(self, value: Union["up.model.fnode.FNode", int, Fraction]):
        """
        Sets the `duration interval` for this `action` as the interval `[value, value]`.

        :param value: The `value` set as both edges of this `action's duration`.
        """
        (value_exp,) = self._environment.expression_manager.auto_promote(value)
        self.set_duration_constraint(up.model.timing.FixedDuration(value_exp))

    def set_closed_duration_interval(
        self,
        lower: Union["up.model.fnode.FNode", int, Fraction],
        upper: Union["up.model.fnode.FNode", int, Fraction],
    ):
        """
        Sets the `duration interval` for this `action` as the interval `[lower, upper]`.

        :param lower: The value set as the lower edge of this `action's duration`.
        :param upper: The value set as the upper edge of this `action's duration`.
        """
        lower_exp, upper_exp = self._environment.expression_manager.auto_promote(
            lower, upper
        )
        self.set_duration_constraint(
            up.model.timing.ClosedDurationInterval(lower_exp, upper_exp)
        )

    def set_open_duration_interval(
        self,
        lower: Union["up.model.fnode.FNode", int, Fraction],
        upper: Union["up.model.fnode.FNode", int, Fraction],
    ):
        """
        Sets the `duration interval` for this action as the interval `]lower, upper[`.

        :param lower: The value set as the lower edge of this `action's duration`.
        :param upper: The value set as the upper edge of this `action's duration`.

        Note that `lower` and `upper` are not part of the interval.
        """
        lower_exp, upper_exp = self._environment.expression_manager.auto_promote(
            lower, upper
        )
        self.set_duration_constraint(
            up.model.timing.OpenDurationInterval(lower_exp, upper_exp)
        )

    def set_left_open_duration_interval(
        self,
        lower: Union["up.model.fnode.FNode", int, Fraction],
        upper: Union["up.model.fnode.FNode", int, Fraction],
    ):
        """
        Sets the `duration interval` for this `action` as the interval `]lower, upper]`.

        :param lower: The value set as the lower edge of this `action's duration`.
        :param upper: The value set as the upper edge of this `action's duration`.

        Note that `lower` is not part of the interval.
        """
        lower_exp, upper_exp = self._environment.expression_manager.auto_promote(
            lower, upper
        )
        self.set_duration_constraint(
            up.model.timing.LeftOpenDurationInterval(lower_exp, upper_exp)
        )

    def set_right_open_duration_interval(
        self,
        lower: Union["up.model.fnode.FNode", int, Fraction],
        upper: Union["up.model.fnode.FNode", int, Fraction],
    ):
        """
        Sets the `duration interval` for this `action` as the interval `[lower, upper[`.

        :param lower: The value set as the lower edge of this `action's duration`.
        :param upper: The value set as the upper edge of this `action's duration`.

        Note that `upper` is not part of the interval.
        """
        lower_exp, upper_exp = self._environment.expression_manager.auto_promote(
            lower, upper
        )
        self.set_duration_constraint(
            up.model.timing.RightOpenDurationInterval(lower_exp, upper_exp)
        )

    def is_conditional(self) -> bool:
        """Returns `True` if the `action` has `conditional effects`, `False` otherwise."""
        # re-implemenation needed for inheritance, delegate implementation.
        return TimedCondsEffs.is_conditional(self)


class SensingAction(InstantaneousAction):
    """This class represents a sensing action."""

    def __init__(
        self,
        _name: str,
        _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
        _env: Optional[Environment] = None,
        **kwargs: "up.model.types.Type",
    ):
        InstantaneousAction.__init__(self, _name, _parameters, _env, **kwargs)
        self._observed_fluents: List["up.model.fnode.FNode"] = []

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, SensingAction):
            return super().__eq__(oth) and set(self._observed_fluents) == set(
                oth._observed_fluents
            )
        else:
            return False

    def __hash__(self) -> int:
        res = super().__hash__()
        for of in self._observed_fluents:
            res += hash(of)
        return res

    def clone(self):
        new_params = OrderedDict()
        for param_name, param in self._parameters.items():
            new_params[param_name] = param.type
        new_sensing_action = SensingAction(self._name, new_params, self._environment)
        new_sensing_action._preconditions = self._preconditions[:]
        new_sensing_action._effects = [e.clone() for e in self._effects]
        new_sensing_action._fluents_assigned = self._fluents_assigned.copy()
        new_sensing_action._fluents_inc_dec = self._fluents_inc_dec.copy()
        new_sensing_action._simulated_effect = self._simulated_effect
        new_sensing_action._observed_fluents = self._observed_fluents.copy()
        return new_sensing_action

    def add_observed_fluents(self, observed_fluents: List["up.model.fnode.FNode"]):
        """
        Adds the given list of observed fluents.

        :param observed_fluents: The list of observed fluents that must be added.
        """
        for of in observed_fluents:
            self.add_observed_fluent(of)

    def add_observed_fluent(self, observed_fluent: "up.model.fnode.FNode"):
        """
        Adds the given observed fluent.

        :param observed_fluent: The observed fluent that must be added.
        """
        self._observed_fluents.append(observed_fluent)

    @property
    def observed_fluents(self) -> List["up.model.fnode.FNode"]:
        """Returns the `list` observed fluents."""
        return self._observed_fluents

    def __repr__(self) -> str:
        b = InstantaneousAction.__repr__(self)[0:-3]
        s = ["sensing-", b]
        s.append("    observations = [\n")
        for e in self._observed_fluents:
            s.append(f"      {str(e)}\n")
        s.append("    ]\n")
        s.append("  }")
        return "".join(s)


class ProbabilisticAction(InstantaneousAction):
    """Represents an action with probabilistic effect."""
    def __init__(
            self,
            _name: str,
            _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
            _env: Optional[Environment] = None,
            **kwargs: "up.model.types.Type",
    ):
        InstantaneousAction.__init__(self, _name, _parameters, _env, **kwargs)
        self._probabilistic_effects: List[up.model.effect.ProbabilisticEffect] = []

    def __repr__(self) -> str:
        b = InstantaneousAction.__repr__(self)[0:-3]
        s = ["ProbabilisticAction-", b]
        s.append("    probabilistic effects = [\n")
        for pe in self._probabilistic_effects:
            s.append(f"      {str(pe)}\n")
        s.append("    ]\n")
        s.append("  }")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, ProbabilisticAction):
            super().__eq__(oth) and set(self._probabilistic_effects) == set(
                oth._probabilistic_effects)
        else:
            return False

    def __hash__(self) -> int:
        res = super().__hash__()
        for pe in self._probabilistic_effects:
            res += hash(pe)
        return res

    def clone(self):
        new_params = OrderedDict()
        for param_name, param in self._parameters.items():
            new_params[param_name] = param.type
        new_probabilistic_action = ProbabilisticAction(self._name, new_params, self._environment)
        new_probabilistic_action._preconditions = self._preconditions[:]
        new_probabilistic_action._effects = [e.clone() for e in self._effects]
        new_probabilistic_action._fluents_assigned = self._fluents_assigned.copy()
        new_probabilistic_action._fluents_inc_dec = self._fluents_inc_dec.copy()
        new_probabilistic_action._simulated_effect = self._simulated_effect
        new_probabilistic_action._probabilistic_effects = [pe.clone() for pe in self._probabilistic_effects]
        return new_probabilistic_action

    def add_probabilistic_effect(
            self,
            fluents: List["up.model.fnode.FNode"],
            probability_func: Callable[
                [
                    "up.model.problem.AbstractProblem",
                    "up.model.state.ROState",
                ],
                List["up.model.fnode.FNode"],
            ]
    ):
        """
        Adds the given `assignment` to the `action's probabilistic_effects`.

        :param fluents: The `fluents` of which `value` is modified by the `assignment`.
        :param probability_func: based on the probability function a value is chosen from the values param
        """


        fluents_exp = self._environment.expression_manager.auto_promote(fluents)

        for f in fluents_exp:
            if not f.is_fluent_exp():
                raise UPUsageError(
                    "fluent field of add_effect must be a Fluent or a FluentExp"
                )

        self._add_probabilistic_effect_instance(
            up.model.effect.ProbabilisticEffect(fluents_exp, probability_func)
        )


    def _add_probabilistic_effect_instance(self, probabilistic_effect: "up.model.effect.ProbabilisticEffect"):
        assert (
                probabilistic_effect.environment() == self._environment
        ), "effect does not have the same environment of the action"
        up.model.effect.check_conflicting_probabilistic_effects(
            probabilistic_effect,
            None,
            self._simulated_effect,
            self._fluents_assigned,
            self._fluents_inc_dec,
            self._probabilistic_effects,
            self._effects,
            "action",
        )
        self._probabilistic_effects.append(probabilistic_effect)
    def probabilistic_effect(self) -> List["up.model.effect.ProbabilisticEffect"]:
        """Returns the `action` `probabilistic effect`."""
        return self._probabilistic_effects

    def set_probabilistic_effect(self, probabilistic_effect: "up.model.effect.ProbabilisticEffect"):
        """
        Sets the given `probabilistic effect` as the only `action's probabilistic effect`.

        :param probabilistic_effect: The `ProbabilisticEffect` instance that must be set as this `action`'s only
            `probabilistic effect`.
        """
        self._probabilistic_effects = [probabilistic_effect]

    def clear_effects(self):
        """Removes all the `Action's effects`."""
        super().clear_effects()
        self._probabilistic_effects = []


class FixDurationStartAction(InstantaneousAction):
    """Represents a start action with fix duration.
    This is the start action of the DurationProbabilisticAction action class
    _end_action - the end action of this start action """
    def __init__(
            self,
            _name: str,
            _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
            _env: Optional[Environment] = None,
            **kwargs: "up.model.types.Type",
    ):
        InstantaneousAction.__init__(self, _name, _parameters, _env, **kwargs)
        self._duration: "up.model.timing.DurationInterval" = (
            up.model.timing.FixedDuration(self._environment.expression_manager.Int(0)))
        self._end_action: ProbabilisticAction = None
    def __repr__(self) -> str:
        b = InstantaneousAction.__repr__(self)[0:-3]
        s = ["FixDurationStartAction-", b]
        s.append(f"    duration = {str(self._duration)}\n")
        s.append(f" end action = {self._end_action.name}")
        s.append("  }")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, FixDurationStartAction):
            super().__eq__(oth) and \
            self._duration == oth._duration and \
                self._end_action == oth._end_action
        else:
            return False

    def __hash__(self) -> int:
        return super().__hash__() + hash(self._duration) + self._end_action.__hash__()

    def clone(self):
        new_params = OrderedDict()
        for param_name, param in self._parameters.items():
            new_params[param_name] = param.type
        new_duration_start_action = FixDurationStartAction(self._name, new_params, self._environment)
        new_duration_start_action._preconditions = self._preconditions[:]
        new_duration_start_action._effects = [e.clone() for e in self._effects]
        new_duration_start_action._fluents_assigned = self._fluents_assigned.copy()
        new_duration_start_action._fluents_inc_dec = self._fluents_inc_dec.copy()
        new_duration_start_action._simulated_effect = self._simulated_effect
        new_duration_start_action._duration = self._duration
        new_duration_start_action._end_action = self._end_action.clone()

        return new_duration_start_action

    def duration(self) -> "up.model.timing.DurationInterval":
        """Returns the `action` `duration interval`."""
        return self._duration

    def set_fixed_duration(self, value: Union["up.model.fnode.FNode", int, Fraction]):
        """
        Sets the `duration interval` for this `action` as the interval `[value, value]`.

        :param value: The `value` set as both edges of this `action's duration`.
        """
        (value_exp,) = self._environment.expression_manager.auto_promote(value)
        self.set_duration_constraint(up.model.timing.FixedDuration(value_exp))

    def set_end_action(self, end_action: ProbabilisticAction):
        """Returns the `end_action` `name`."""
        self._end_action = end_action

    def end_action(self) -> ProbabilisticAction:
        """Returns the `end_action` `name`."""
        return self._end_action


class DurationProbabilisticAction(Action):
    """Represents an action with fix duration and probabilistic effect.
    The action is compound from instantaneous two actions -
    start action:
        preconditions - the precondition of the action
        effects - during activation effects
    end action:
        preconditions - start action is activated
        effects - the effects of the action

    **** don't support conditional effect and simulated effect
    """

    def __init__(
            self,
            _name: str,
            _parameters: Optional["OrderedDict[str, up.model.types.Type]"] = None,
            _env: Optional[Environment] = None,
            **kwargs: "up.model.types.Type",
    ):
        Action.__init__(self, _name, _parameters, _env, **kwargs)
        self._start_action = FixDurationStartAction("start_"+_name, _parameters)
        self._end_action = ProbabilisticAction("end_"+_name, _parameters)
        self._start_action.set_end_action(self._end_action)

    def __repr__(self) -> str:
        b = Action.__repr__(self)[0:-3]
        s = ["duration probabilistic-", b]
        s.append(f"    start action = {self._start_action.__repr__()}\n")
        s.append(f"    end action = {self._end_action.__repr__()}\n")
        s.append("  }")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, DurationProbabilisticAction):
            super().__eq__(oth) and self._start_action == oth._start_action and self._end_action == oth._end_action
        else:
            return False

    def __hash__(self) -> int:
        return super().__hash__() + hash(self._start_action) + hash(self._end_action)

    def clone(self):
        new_params = OrderedDict()
        for param_name, param in self._parameters.items():
            new_params[param_name] = param.type
        new_durative_probabilistic_action = DurationProbabilisticAction(self._name, new_params, self._environment)
        new_durative_probabilistic_action._start_action = self._start_action.clone()
        new_durative_probabilistic_action._end_action = self._end_action.clone()
        return new_durative_probabilistic_action

    def start_action(self) -> FixDurationStartAction:
        return self._start_action

    def end_action(self) -> ProbabilisticAction:
        return self._end_action

    def preconditions(self) -> List["up.model.fnode.FNode"]:
        """Returns the `list` of the `Action` `preconditions`."""
        return self._start_action.preconditions()

    def clear_preconditions(self):
        """Removes all the `Action preconditions`"""
        self._start_action.clear_preconditions()

    def during_activation_effects(self) -> List["up.model.effect.Effect"]:
        """Returns the `list` of the `Action effects`."""
        return self._start_action.effects()

    def clear_during_activation_effects(self):
        """Removes all the `Action's effects`."""
        self._start_action.clear_effects()

    def effects(self) -> List["up.model.effect.Effect"]:
        """Returns the `list` of the `Action effects`."""
        return self._end_action.effects()

    def clear_effects(self):
        """Removes all the `Action's effects`."""
        self._end_action.clear_effects()

    def probabilistic_effect(self) -> List["up.model.effect.ProbabilisticEffect"]:
        """Returns the `action` `probabilistic effect`."""
        return self._end_action.probabilistic_effects()

    def add_precondition(
        self,
        precondition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool,
        ],
    ):
        """
        Adds the given expression to `action's preconditions`.

        :param precondition: The expression that must be added to the `action's preconditions`.
        """
        self._start_action.add_precondition(precondition)

    def add_during_activation_effect(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
    ):
        """
        Adds the given `assignment` to the `action's duration effects`.

        :param fluent: The `fluent` of which `value` is modified by the `assignment`.
        :param value: The `value` to assign to the given `fluent`.
        :param condition: The `condition` in which this `duration effect` is applied; the default
            value is `True`.
        """

        self._start_action.add_effect(fluent, value, condition)

    def add_end_precondition(
            self,
            precondition: Union[
                "up.model.fnode.FNode",
                "up.model.fluent.Fluent",
                "up.model.parameter.Parameter",
                bool,
            ],
    ):
        """
        Adds the given expression to `action's preconditions`.

        :param precondition: The expression that must be added to the `action's preconditions`.
        """
        self._end_action.add_precondition(precondition)
    def add_effect(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
    ):
        """
        Adds the given `assignment` to the `action's effects`.

        :param fluent: The `fluent` of which `value` is modified by the `assignment`.
        :param value: The `value` to assign to the given `fluent`.
        :param condition: The `condition` in which this `effect` is applied; the default
            value is `True`.
        """

        self._end_action.add_effect(fluent, value, condition)

    def add_probabilistic_effect(
            self,
            fluents: List["up.model.fnode.FNode"],
            probability_func: Callable[
                [
                    "up.model.problem.AbstractProblem",
                    "up.model.state.ROState",
                ],
                List["up.model.fnode.FNode"],
            ]
    ):
        """
        Adds the given `assignment` to the `action's probabilistic_effects`.

        :param fluents: The `fluent` of which `value` is modified by the `assignment`.
        :param probability_func: based on the probability function a value is chosen from the values param
        """
        self._end_action.add_probabilistic_effect(fluents, probability_func)

    def set_probabilistic_effect(self, probabilistic_effect: "up.model.effect.ProbabilisticEffect"):
        """
        Sets the given `probabilistic effect` as the only `action's probabilistic effect`.

        :param probabilistic_effect: The `ProbabilisticEffect` instance that must be set as this `action`'s only
            `probabilistic effect`.
        """
        self._end_action.set_probabilistic_effect(probabilistic_effect)

    def _set_preconditions(self, preconditions: List["up.model.fnode.FNode"]):
        self._start_action._set_preconditions(preconditions)

    def duration(self) -> "up.model.timing.DurationInterval":
        """Returns the `action` `duration interval`."""
        return self._start_action.duration()

    def set_fixed_duration(self, value: Union["up.model.fnode.FNode", int, Fraction]):
        """
        Sets the `duration interval` for this `action` as the interval `[value, value]`.

        :param value: The `value` set as both edges of this `action's duration`.
        """
        self._start_action.set_fixed_duration(value)


def start_end_actions(problem, actions: List["up.model.action.DurationProbabilisticAction"]):
    action = up.shortcuts.UserType('action')
    action_occurs = up.model.Fluent('action_occurs', up.shortcuts.BoolType(), a=action)

    problem.add_fluent(action)
    problem.add_fluent(action_occurs, default_initial_value=False)

    for a in actions:
        start_a = up.model.Object("start_" + a.name, action)

        a.add_during_activation_effect(action_occurs(start_a), True)
        a.add_end_precondition(action_occurs(start_a), True)
        a.add_effect(action_occurs(start_a), False)
        problem.add_object(start_a)


