from unified_planning.shortcuts import *
from scipy.stats import bernoulli


def add_object_condition_effect(action, object):
    action.add_precondition(object)
    action.add_during_activation_effect(object, False)
    action.add_effect(object, True)


duration_probabilistic_actions = []
problem = unified_planning.model.Problem('stuck_car')

""" Init things that can be pushed """
Car = UserType('Car')
car = unified_planning.model.Object('car', Car)
problem.add_object(car)

GasPedal = UserType('GasPedal')
gasPedal = unified_planning.model.Object('gasPedal', GasPedal)
problem.add_object(gasPedal)

""" Init rocks """
Rock = UserType('Rock')
rocks_names = ['bad', 'good']
rocks = [unified_planning.model.Object(r, Rock) for r in rocks_names]
problem.add_objects(rocks)

""" Init body parts - 
    when performing an action at least one of the body parts will be occupied 
"""
BodyPart = UserType('BodyPart')
bodyParts_names = ['hands', 'legs']
bodyParts = [unified_planning.model.Object(b, BodyPart) for b in bodyParts_names]
problem.add_objects(bodyParts)

car_out = unified_planning.model.Fluent('car_out', BoolType())
problem.set_initial_value(car_out, False)

tired = unified_planning.model.Fluent('tired', BoolType())
problem.set_initial_value(tired, True)

got_rock = unified_planning.model.Fluent('got_rock', BoolType(), r=Rock)
problem.add_fluent(got_rock, default_initial_value=False)

free = unified_planning.model.Fluent('free', BoolType(), b=BodyPart)
problem.add_fluent(free, default_initial_value=True)

rock_under_car = unified_planning.model.Fluent('rock_under_car', BoolType(), r=Rock)
problem.add_fluent(rock_under_car, default_initial_value=False)

""" Actions """

""" Rest Action """
rest = unified_planning.model.DurativeAction('rest')
rest.add_condition(free(bodyParts[0]), True)
rest.add_condition(free(bodyParts[1]), True)
rest.set_fixed_duration(1)
rest.add_effect(EndTiming(), tired, False)
problem.add_action(rest)

""" Place a rock under the car Action """
place_rock = unified_planning.model.DurationProbabilisticAction('place_rock', rock=Rock)
duration_probabilistic_actions.append(place_rock)
rock = place_rock.parameter('rock')
place_rock.set_fixed_duration(3)
place_rock.add_precondition(got_rock(rock))
add_object_condition_effect(place_rock, free(bodyParts[0]))
add_object_condition_effect(place_rock, free(bodyParts[1]))


def tired_place_probability(problem, state):
    # The probability of finding a good rock when searching
    p = 0.8
    rv = bernoulli(p)
    tired = rv.rvs(1)[0][0]
    if tired:
        return [True]
    return [False]


place_rock.add_effect(rock_under_car(rock), True)
place_rock.add_effect(got_rock(rock), False)
place_rock.add_probabilistic_effect([tired], tired_place_probability)
problem.add_action(place_rock)

""" Search a rock Action 
    the robot can find a one of the rocks"""
search = unified_planning.model.action.DurationProbabilisticAction('search')
duration_probabilistic_actions.append(search)
search.set_fixed_duration(3)
add_object_condition_effect(search, free(bodyParts[0]))
add_object_condition_effect(search, free(bodyParts[1]))


def rock_probability(problem, state):
    # The probability of finding a good rock when searching
    p = 0.8
    rv = bernoulli(p)
    rock = rv.rvs(1)[0][0]
    rock_found = [False, False]
    rock_found[rock] = True
    return rock_found


search.add_probabilistic_effect([got_rock(rocks[0]), got_rock(rocks[1])], rock_probability)
problem.add_action(search)

""" Push Actions """

""" Push Gas Pedal Action """
push_gas = unified_planning.model.action.DurationProbabilisticAction('push_gas')
duration_probabilistic_actions.append(push_gas)
push_gas.set_fixed_duration(2)
add_object_condition_effect(push_gas, free(bodyParts[1]))

problem.add_action(push_gas)

""" Push Car Action """
push_car = unified_planning.model.action.DurationProbabilisticAction('push_car')
duration_probabilistic_actions.append(push_car)
push_car.set_fixed_duration(2)
add_object_condition_effect(push_car, free(bodyParts[0]))

problem.add_action(push_car)

action_occurs, durative_probabilistic_action_objects = unified_planning.model.action.start_end_actions(problem,
                                                                                                       duration_probabilistic_actions)
start_push_gas, start_push_car = (None, None)
for o in durative_probabilistic_action_objects:
    if o.name == 'start-push_gas':
        start_push_gas = o
    if o.name == 'start-push_car':
        start_push_car = o


def tired_push_probability(problem, state):
    # The probability of finding a good rock when searching
    p = 0.8
    rv = bernoulli(p)
    tired = rv.rvs(1)[0][0]
    if tired:
        return [True]
    return [False]


def push_probability(problem, state):
    # The probability of getting the car out when pushing
    p = 0
    predicates = state.predicates

    # The bad rock is under the car
    if rock_under_car(rocks[0]) in predicates:
        if action_occurs(start_push_car) in predicates and action_occurs(start_push_gas) in predicates:
            p = 0.8
        elif action_occurs(start_push_car) in predicates:
            p = 0.8
        elif action_occurs(start_push_gas) in predicates:
            p = 0.8

    # The good rock is under the car
    elif rock_under_car(rocks[1]) in predicates:
        if action_occurs(start_push_car) in predicates and action_occurs(start_push_gas) in predicates:
            p = 0.8
        elif action_occurs(start_push_car) in predicates:
            p = 0.8
        elif action_occurs(start_push_gas) in predicates:
            p = 0.8

    # There isn't a rock under the car
    elif action_occurs(start_push_car) in predicates and action_occurs(start_push_gas) in predicates:
        p = 0.8
    elif action_occurs(start_push_car) in predicates:
        p = 0.8
    elif action_occurs(start_push_gas) in predicates:
        p = 0.8

    rv = bernoulli(p)
    out = rv.rvs(1)[0][0]
    if out:
        return [True]
    return [False]


push_gas.add_probabilistic_effect([car_out], push_probability)
push_car.add_probabilistic_effect([car_out], push_probability)
push_car.add_probabilistic_effect([tired], tired_push_probability)

deadline = Timing(delay=6, timepoint=Timepoint(TimepointKind.START))
problem.add_timed_goal(deadline, car_out)

print(problem)
