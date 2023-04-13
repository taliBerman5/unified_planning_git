from unified_planning.shortcuts import *
from scipy.stats import bernoulli

problem = unified_planning.model.Problem('stuck_car')

""" Init things that can be pushed """
Object = UserType('Object')
objects_names = ['gasPedal', 'car']
objects = [unified_planning.model.Object(o, Object) for o in objects_names]
problem.add_objects(objects)

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

car_stuck = unified_planning.model.Fluent('car_stuck', BoolType())
problem.add_fluent(car_stuck, default_initial_value=True)

tired = unified_planning.model.Fluent('tired', BoolType())
problem.add_fluent(tired, default_initial_value=False)

pushed = unified_planning.model.Fluent('pushed', BoolType(), o=Object)
problem.add_fluent(pushed, default_initial_value=False)

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
rock = place_rock.parameter('rock')


def tired_place_probability(problem, state):
    # The probability of finding a good rock when searching
    p = 0.8
    rv = bernoulli(p)
    tired = rv.rvs(1)[0][0]
    if tired:
        return [True]
    return [False]


place_rock.add_precondition(got_rock(rock))
place_rock.add_precondition(free(bodyParts[0]))
place_rock.add_precondition(free(bodyParts[1]))
place_rock.set_fixed_duration(3)
place_rock.add_during_activation_effect(free(bodyParts[0]), False)
place_rock.add_during_activation_effect(free(bodyParts[1]), False)
place_rock.add_effect(rock_under_car(rock), True)
place_rock.add_effect(free(bodyParts[0]), True)
place_rock.add_effect(free(bodyParts[1]), True)
place_rock.add_probabilistic_effect([tired], tired_place_probability)
problem.add_action(place_rock)


""" Search a rock Action 
    the robot can find a one of the rocks"""
search = unified_planning.model.action.DurationProbabilisticAction('search')
search.add_precondition(free(bodyParts[0]))
search.add_precondition(free(bodyParts[1]))


def rock_probability(problem, state):
    # The probability of finding a good rock when searching
    p = 0.8
    rv = bernoulli(p)
    rock = rv.rvs(1)[0][0]
    rock_found = [False, False]
    rock_found[rock] = True
    return rock_found


search.add_probabilistic_effect([got_rock(rocks[0]), got_rock(rock[1])], rock_probability)
search.add_during_activation_effect(free(bodyParts[0]), False)
search.add_during_activation_effect(free(bodyParts[1]), False)
search.add_effect(free(bodyParts[0]), True)
search.add_effect(free(bodyParts[1]), True)
problem.add_action(search)

push = unified_planning.model.action.DurationProbabilisticAction('push', object=Object)

problem.add_action(push)

problem.add_goal(car_stuck)
print(problem)
