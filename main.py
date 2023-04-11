import unified_planning
from unified_planning.model.action import ProbabilisticAction
from unified_planning.shortcuts import *

Match = UserType('Match')
Fuse = UserType('Fuse')

handfree = Fluent('handfree')
light = Fluent('light')
match_used = Fluent('match_used', BoolType(), m=Match)
fuse_mended = Fluent('fuse_mended', BoolType(), f=Fuse)

problem = Problem('MatchCellar')

problem.add_fluent(handfree)
problem.add_fluent(light)
problem.add_fluent(match_used, default_initial_value=False)
problem.add_fluent(fuse_mended, default_initial_value=False)

problem.set_initial_value(light, False)
problem.set_initial_value(handfree, True)

fuses = [Object(f'f{i}', Fuse) for i in range(3)]
matches = [Object(f'm{i}', Match) for i in range(3)]

problem.add_objects(fuses)
problem.add_objects(matches)



action = ProbabilisticAction('light_match', m=Match)
def fun(problem, state):
    return 0
action.add_probabilistic_effect(handfree,fun , [True, False])
print(action)
#

#
#
# Location = UserType('Location')
# Robot = UserType('Robot')
#
# at = Fluent('at', Location, robot=Robot)
# battery_charge = Fluent('battery_charge', IntType(0, 100), robot=Robot)
#
#
# move = ProbabilisticAction('move', robot=Robot, l_from=Location, l_to=Location)
# robot = move.parameter('robot')
# l_from = move.parameter('l_from')
# l_to = move.parameter('l_to')
# move.add_precondition(Equals(at(robot), l_from))
# move.add_precondition(GE(battery_charge(robot), 10))
# move.add_precondition(Not(Equals(l_from, l_to)))
# move.add_effect(at(robot), l_to)
# def fun(problem, state):
#     return 0
#
#
# l1 = Object('l1', Location)
# l2 = Object('l2', Location)
# r1 = Object('r1', Robot)
#
# problem = Problem('robot_with_simulated_effects')
# problem.add_fluent(at)
# problem.add_fluent(battery_charge)
# problem.add_action(move)
# problem.add_object(l1)
# problem.add_object(l2)
# problem.add_object(r1)
#
#
# move.add_probabilistic_effect([at(robot)], fun, [l1, l2])
# print(move)




Location = UserType('Location')
robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
connected = unified_planning.model.Fluent('connected', BoolType(), l_from=Location, l_to=Location)
move = unified_planning.model.InstantaneousAction('move', l_from=Location, l_to=Location)
l_from = move.parameter('l_from')
l_to = move.parameter('l_to')
move.add_precondition(connected(l_from, l_to))
move.add_precondition(robot_at(l_from))
move.add_effect(robot_at(l_from), False)
move.add_effect(robot_at(l_to), True)
problem = unified_planning.model.Problem('robot')
problem.add_fluent(robot_at, default_initial_value=False)
problem.add_fluent(connected, default_initial_value=False)
problem.add_action(move)
NLOC = 10
locations = [unified_planning.model.Object('l%s' % i, Location) for i in range(NLOC)]
problem.add_objects(locations)
problem.set_initial_value(robot_at(locations[0]), True)
for i in range(NLOC - 1):
    problem.set_initial_value(connected(locations[i], locations[i+1]), True)

problem.add_goal(robot_at(locations[-1]))


with OneshotPlanner(name='pyperplan') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("Pyperplan returned: %s" % result.plan)
    else:
        print("No plan found.")


#
# with OneshotPlanner(problem_kind=problem.kind) as planner:
#     result = planner.solve(problem)
#     plan = result.plan
#     if plan is not None:
#         print("%s returned:" % planner.name)
#         for start, action, duration in plan.timed_actions:
#             print("%s: %s [%s]" % (float(start), action, float(duration)))
#     else:
#         print("No plan found.")