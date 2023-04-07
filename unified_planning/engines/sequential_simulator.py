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


from fractions import Fraction
from warnings import warn
import unified_planning as up
from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.engines.engine import Engine
from unified_planning.engines.mixins.simulator import Event, SimulatorMixin
from unified_planning.exceptions import UPUsageError, UPConflictingEffectsException
from unified_planning.plans import ActionInstance
from unified_planning.model import FNode, Type, ExpressionManager
from unified_planning.model.types import _RealType
from unified_planning.model.walkers import StateEvaluator
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, cast


class InstantaneousEvent(Event):
    """Implements the Event class for an Instantaneous Action."""

    def __init__(
        self,
        conditions: List["up.model.FNode"],
        effects: List["up.model.Effect"],
        simulated_effect: Optional["up.model.SimulatedEffect"] = None,
    ):
        self._conditions = conditions
        self._effects = effects
        self._simulated_effect = simulated_effect

    @property
    def conditions(self) -> List["up.model.FNode"]:
        return self._conditions

    @property
    def effects(self) -> List["up.model.Effect"]:
        return self._effects

    @property
    def simulated_effect(self) -> Optional["up.model.SimulatedEffect"]:
        return self._simulated_effect


class SequentialSimulator(Engine, SimulatorMixin):
    """
    Sequential SimulatorMixin implementation.

    This Simulator, when considering if a state is goal or not, ignores the
    quality metrics.
    """

    def __init__(
        self, problem: "up.model.Problem", error_on_failed_checks: bool = True, **kwargs
    ):
        Engine.__init__(self)
        self.error_on_failed_checks = error_on_failed_checks
        SimulatorMixin.__init__(self, problem)
        pk = problem.kind
        if not Grounder.supports(pk):
            msg = f"The Grounder used in the {type(self)} does not support the given problem"
            if self.error_on_failed_checks:
                raise UPUsageError(msg)
            else:
                warn(msg)
        assert isinstance(self._problem, up.model.Problem)
        self._grounder = GrounderHelper(problem)
        self._actions = set(self._problem.actions)
        self._events: Dict[
            Tuple["up.model.Action", Tuple["up.model.FNode", ...]], List[Event]
        ] = {}
        self._se = StateEvaluator(self._problem)
        self._all_events_grounded: bool = False

    def _get_unsatisfied_conditions(
        self, event: "Event", state: "up.model.ROState", early_termination: bool = False
    ) -> List["up.model.FNode"]:
        """
        Returns the list of unsatisfied event conditions evaluated in the given state.
        If the flag `early_termination` is set, the method ends and returns at the first unsatisfied condition.

        :param state: The `State` in which the event conditions are evaluated.
        :param early_termination: Flag deciding if the method ends and returns at the first unsatisfied condition.
        :return: The list of all the event conditions that evaluated to `False` or the list containing the first
            condition evaluated to False if the flag `early_termination` is set.
        """
        unsatisfied_conditions = []
        for c in event.conditions:
            evaluated_cond = self._se.evaluate(c, state)
            if (
                not evaluated_cond.is_bool_constant()
                or not evaluated_cond.bool_constant_value()
            ):
                unsatisfied_conditions.append(c)
                if early_termination:
                    break

        # check that the assignments will respect the bound typing
        new_bounded_types_values: Dict["up.model.FNode", "up.model.FNode"] = {}
        assigned_fluent: Set["up.model.FNode"] = set()
        em = self._problem.environment.expression_manager
        for effect in event.effects:
            lower_bound, upper_bound = None, None
            f_type = cast(_RealType, effect.fluent.type)
            if f_type.is_int_type() or f_type.is_real_type():
                lower_bound, upper_bound = f_type.lower_bound, f_type.upper_bound
            if lower_bound is not None or upper_bound is not None:
                fluent, value = self._evaluate_effect(
                    effect, state, new_bounded_types_values, assigned_fluent, em
                )
                if fluent is not None:
                    assert value is not None
                    new_bounded_types_values[fluent] = value
                    if lower_bound is not None and lower_bound > cast(
                        Fraction, value.constant_value()
                    ):
                        unsatisfied_conditions.append(em.LE(lower_bound, fluent))
                        if early_termination:
                            break
                    if upper_bound is not None and upper_bound < cast(
                        Fraction, value.constant_value()
                    ):
                        unsatisfied_conditions.append(em.LE(fluent, upper_bound))
                        if early_termination:
                            break
        if event.simulated_effect is not None:
            to_check = False
            for f in event.simulated_effect.fluents:
                f_type = cast(_RealType, f.type)
                if (f_type.is_int_type() or f_type.is_real_type()) and (
                    f_type.lower_bound is not None or f_type.upper_bound is not None
                ):
                    to_check = True
                    break
            if to_check:
                for f, v in zip(
                    event.simulated_effect.fluents,
                    event.simulated_effect.function(self._problem, state, {}),
                ):
                    lower_bound, upper_bound = None, None
                    f_type = cast(_RealType, f.type)
                    if f_type.is_int_type() or f_type.is_real_type():
                        lower_bound, upper_bound = (
                            f_type.lower_bound,
                            f_type.upper_bound,
                        )
                    if lower_bound is not None or upper_bound is not None:
                        if (
                            lower_bound is not None
                            and cast(Fraction, v.constant_value()) < lower_bound
                        ):
                            unsatisfied_conditions.append(em.LE(lower_bound, f))
                            if early_termination:
                                break
                        if (
                            upper_bound is not None
                            and cast(Fraction, v.constant_value()) > upper_bound
                        ):
                            unsatisfied_conditions.append(em.LE(f, upper_bound))
                            if early_termination:
                                break
        return unsatisfied_conditions

    def _apply(
        self, event: "Event", state: "up.model.COWState"
    ) -> Optional["up.model.COWState"]:
        """
        Returns `None` if the event is not applicable in the given state, otherwise returns a new COWState,
        which is a copy of the given state but the applicable effects of the event are applied; therefore
        some fluent values are updated.

        :param state: the state where the event formulas are calculated.
        :param event: the event that has the information about the conditions to check and the effects to apply.
        :return: None if the event is not applicable in the given state, a new COWState with some updated values
            if the event is applicable.
        """
        if not self.is_applicable(event, state):
            return None
        else:
            return self.apply_unsafe(event, state)

    def _apply_unsafe(
        self, event: "Event", state: "up.model.COWState"
    ) -> "up.model.COWState":
        """
        Returns a new COWState, which is a copy of the given state but the applicable effects of the event are applied; therefore
        some fluent values are updated.
        IMPORTANT NOTE: Assumes that self.is_applicable(state, event) returns True

        :param state: the state where the event formulas are evaluated.
        :param event: the event that has the information about the effects to apply.
        :return: A new COWState with some updated values.
        """
        updated_values: Dict["up.model.FNode", "up.model.FNode"] = {}
        assigned_fluent: Set["up.model.FNode"] = set()
        em = self._problem.environment.expression_manager
        for effect in event.effects:
            fluent, value = self._evaluate_effect(
                effect, state, updated_values, assigned_fluent, em
            )
            if fluent is not None:
                assert value is not None
                updated_values[fluent] = value
        if event.simulated_effect is not None:
            for f, v in zip(
                event.simulated_effect.fluents,
                event.simulated_effect.function(self._problem, state, {}),
            ):
                old_value = updated_values.get(f, None)
                # If f was already modified and it was modified by an increase/decrease or with an assign
                # with a different value
                if old_value is not None and (
                    f not in assigned_fluent
                    or old_value.constant_value() != v.constant_value()
                ):
                    if not f.type.is_bool_type():
                        raise UPConflictingEffectsException(
                            f"The fluent {f} is modified with different values in the same event."
                        )
                    # solve with add-after-delete logic
                    elif not old_value.bool_constant_value():
                        updated_values[f] = v
                else:
                    updated_values[f] = v
        return state.make_child(updated_values)

    def _evaluate_effect(
        self,
        effect: "up.model.Effect",
        state: "up.model.ROState",
        updated_values: Dict["up.model.FNode", "up.model.FNode"],
        assigned_fluent: Set["up.model.FNode"],
        em: ExpressionManager,
    ) -> Tuple[Optional[FNode], Optional[FNode]]:
        """
        Evaluates the given effect in the state, and returns the fluent affected
        by this effect and the new value that is assigned to the fluent.

        If the effect is conditional and the condition evaluates to False in the state,
        (None, None) is returned.

        :param effect: The effect to evaluate.
        :param state: The state in which the effect is evaluated.
        :param updated_values: Map from fluents to their value, used to correctly evaluate
            more than one increase/decrease effect on the same fluent.
        :param assigned_fluent: The set containing all the fluents already assigned in the
            event containing this effect.
        :param em: The current environment expression manager.
        :return: The Tuple[Fluent, Value], where the fluent is the one affected by the given
            effect and value is the new value assigned to the fluent.
        :raises UPConflictingEffectsException: If to the same fluent are assigned 2 different
            values.
        """
        evaluated_args = tuple(self._se.evaluate(a, state) for a in effect.fluent.args)
        fluent = self._problem.environment.expression_manager.FluentExp(
            effect.fluent.fluent(), evaluated_args
        )
        if (not effect.is_conditional()) or self._se.evaluate(
            effect.condition, state
        ).is_true():
            new_value = self._se.evaluate(effect.value, state)
            if effect.is_assignment():
                old_value = updated_values.get(fluent, None)
                if (
                    old_value is not None
                    and new_value.constant_value() != old_value.constant_value()
                ):
                    if not fluent.type.is_bool_type():
                        raise UPConflictingEffectsException(
                            f"The fluent {fluent} is modified by 2 different assignments in the same event."
                        )
                    # solve with add-after-delete logic
                    elif not old_value.bool_constant_value():
                        return fluent, new_value
                    else:
                        return None, None
                elif old_value is not None and fluent not in assigned_fluent:
                    raise UPConflictingEffectsException(
                        f"The fluent {fluent} is modified by 1 assignments and an increase/decrease in the same event."
                    )
                else:
                    assigned_fluent.add(fluent)
                    return fluent, new_value
            else:
                if fluent in assigned_fluent:
                    raise UPConflictingEffectsException(
                        f"The fluent {fluent} is modified by an assignment and an increase/decrease in the same event."
                    )
                # If the fluent is in updated_values, we take his modified value, (which was modified by another increase or decrease)
                # otherwise we take it's evaluation in the state as it's value.
                f_eval = updated_values.get(fluent, self._se.evaluate(fluent, state))
                if effect.is_increase():
                    return (
                        fluent,
                        em.auto_promote(
                            f_eval.constant_value() + new_value.constant_value()
                        )[0],
                    )
                elif effect.is_decrease():
                    return (
                        fluent,
                        em.auto_promote(
                            f_eval.constant_value() - new_value.constant_value()
                        )[0],
                    )
                else:
                    raise NotImplementedError
        else:
            return None, None

    def _get_applicable_events(self, state: "up.model.ROState") -> Iterator["Event"]:
        """
        Returns a view over all the events that are applicable in the given State;
        an Event is considered applicable in a given State, when all the Event condition
        simplify as True when evaluated in the State.

        :param state: The state where the formulas are evaluated.
        :return: an Iterator of applicable Events.
        """
        # if the problem was never fully grounded before,
        # ground it and save all the new events. For every event
        # that is applicable, yield it.
        # Otherwise just return all the applicable events
        if not self._all_events_grounded:
            # perform total grounding
            self._all_events_grounded = True
            # for every grounded action, translate it in an Event
            for (
                original_action,
                params,
                grounded_action,
            ) in self._grounder.get_grounded_actions():
                for event in self._get_or_create_events(
                    original_action, params, grounded_action
                ):
                    if self.is_applicable(event, state):
                        yield event
        else:  # the problem has been fully grounded before, just check for event applicability
            for events in self._events.values():
                for event in events:
                    if self.is_applicable(event, state):
                        yield event

    def _get_events(
        self,
        action: "up.model.Action",
        parameters: Union[
            Tuple["up.model.Expression", ...], List["up.model.Expression"]
        ],
    ) -> List["Event"]:
        """
        Returns a list containing all the events derived from the given action, grounded with the given parameters.

        :param action: The action containing the information to create the event.
        :param parameters: The parameters needed to ground the action
        :return: the List of Events derived from this action with these parameters.
        """
        # sanity check
        if action not in self._actions:
            raise UPUsageError(
                "The action given as parameter does not belong to the problem given to the SequentialSimulator."
            )
        params_exp = tuple(
            self._problem.environment.expression_manager.auto_promote(parameters)
        )
        grounded_action = self._grounder.ground_action(action, params_exp)
        event_list = self._get_or_create_events(action, params_exp, grounded_action)
        return event_list

    def _get_unsatisfied_goals(
        self, state: "up.model.ROState", early_termination: bool = False
    ) -> List["up.model.FNode"]:
        """
        Returns the list of unsatisfied goals evaluated in the given state.
        If the flag "early_termination" is set, the method ends and returns at the first unsatisfied goal.

        :param state: The State in which the problem goals are evaluated.
        :param early_termination: Flag deciding if the method ends and returns at the first unsatisfied goal.
        :return: The list of all the goals that evaluated to False or the list containing the first goal evaluated to False if the flag "early_termination" is set.
        """
        unsatisfied_goals = []
        for g in cast(up.model.Problem, self._problem).goals:
            g_eval = self._se.evaluate(g, state).bool_constant_value()
            if not g_eval:
                unsatisfied_goals.append(g)
                if early_termination:
                    break
        return unsatisfied_goals

    @property
    def name(self) -> str:
        return "sequential_simulator"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        supported_kind = up.model.ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_numbers("CONTINUOUS_NUMBERS")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITY")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("TEMPORAL_OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SequentialSimulator.supported_kind()

    def _get_or_create_events(
        self,
        original_action: "up.model.Action",
        params: Tuple["up.model.FNode", ...],
        grounded_action: Optional["up.model.Action"],
    ) -> List[Event]:
        """
        Support function that takes the `original Action`, the `parameters` used to ground the `grounded Action` and
        the `grounded Action` itself, and adds the corresponding `List of Events` to this `Simulator`. If the
        corresponding `Events` were already created, the same value is returned and no new `Events` are created.

        :param original_action: The `Action` of the :class:`~unified_planning.model.Problem` grounded with the given `params`.
        :param params: The expressions used to ground the `original_action`.
        :param grounded_action: The grounded action, result of the `original_action` grounded with the given `parameters`.
        :return: The retrieved or created `List of Events` corresponding to the `grounded_action`.
        """
        if isinstance(original_action, up.model.InstantaneousAction):
            # check if the event is already cached; if not: create it and cache it
            key = (original_action, params)
            event_list = self._events.get(key, None)
            if event_list is None:
                if (
                    grounded_action is None
                ):  # The grounded action is meaningless, no event associated
                    event_list = []
                else:
                    assert isinstance(grounded_action, up.model.InstantaneousAction)
                    event_list = [
                        InstantaneousEvent(
                            grounded_action.preconditions,
                            grounded_action.effects,
                            grounded_action.simulated_effect,
                        )
                    ]
                self._events[key] = event_list
            # sanity check
            assert len(event_list) < 2
            return event_list
        else:
            raise NotImplementedError(
                "The SequentialSimulator currently supports only InstantaneousActions."
            )
