import unified_planning as up
from unified_planning.shortcuts import *


problem = unified_planning.model.Problem('stuck_car')

""" Init things that can be pushed """
Object = UserType('Object')
objects_names = ['gasPedal', 'car']
objects = [unified_planning.model.Object(o, Object) for o in objects_names]
problem.add_objects(objects)


""" Init rocks """
Rock = UserType('Rock')
NRock = 2
rocks = [unified_planning.model.Object('r%s' % i, Rock) for i in range(NRock)]
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
rest = unified_planning.model.DurativeAction('rest')
rest.add_condition(free(bodyParts[0]), True)
rest.add_condition(free(bodyParts[1]), True)
rest.set_fixed_duration(1)
rest.add_effect(EndTiming(), tired, False)
problem.add_action(rest)

place_rock = unified_planning.model.DurativeAction('place_rock', rock=Rock)
rock = place_rock.parameter('rock')
place_rock.add_condition(got_rock(rock), True)
place_rock.add_condition(free(bodyParts[0]), True)
place_rock.add_condition(free(bodyParts[1]), True)
place_rock.set_fixed_duration(3)
place_rock.add_effect(EndTiming(), rock_under_car(rock), True)
problem.add_action(place_rock)


search = up.model.action.DurationProbabilisticAction('search')
search.add_precondition(free(bodyParts[0]))
search.add_precondition(free(bodyParts[1]))
search.add_probabilistic_effect(got_rock)  #TODO

search.add_effect(free(bodyParts[0]), False)
search.add_effect(free(bodyParts[1]), False)

problem.add_action(search)

push = up.model.action.DurationProbabilisticAction('push', object=Object)

problem.add_action(push)


problem.add_goal(car_stuck)
print(problem)


# def fun(problem, state):
#     return 0
# action.add_probabilistic_effect(handfree,fun , [True, False])
# print(action)
#
