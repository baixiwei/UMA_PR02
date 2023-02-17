# -*- coding: utf-8 -*-

from uma import *

### Rules

## General Arithmetic Rules

R_op1_none_to_zero = Rule( # if operand1 is none in an unsolved problem, replace it with zero
    name    = 'op1_none_to_zero',
    goal    = pat_solve_prob,
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operand1'),
                'pattern':  'BLANK'}],
    acts    = [{'action':   'pop_goal'}] +
              make_SFA_acts(Variable('operator'), Number(0), Variable('operand2'), Variable('problem'), 'answer'))

R_op2_none_to_zero = Rule( # if operand2 is none in an unsolved problem, replace it with zero
    name    = 'op2_none_to_zero',
    goal    = pat_solve_prob,
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operand2'),
                'pattern':  'BLANK'}],
    acts    = [{'action':   'pop_goal'}] +
              make_SFA_acts(Variable('operator'), Variable('operand1'), Number(0), Variable('problem'), 'answer'))

R_add_to_zero = Rule( # x plus zero is x
    name    = 'add_to_zero',
    goal    = pat_solve_prob,
    conds   = [# problem is an unsolved addition problem
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operator'),
                'pattern':  '+'},
               # one of the operands, x, is a number and the other is zero
               ('or',
                ('and',
                 {'target':     Variable('operand1'),
                  'pattern':    Number(0)},
                 {'target':     Variable('operand2'),
                  'pattern':    Variable('x')},
                 {'target':     Variable('x'),
                  'pattern':    Number()}),
                ('and',
                 {'target':     Variable('operand2'),
                  'pattern':    Number(0)},
                 {'target':     Variable('operand1'),
                  'pattern':    Variable('x')},
                 {'target':     Variable('x'),
                  'pattern':    Number()}))],
    acts    = [# set problem answer to x and pop goal
               {'action':   'set_feature',
                'chunk':    Variable('problem'),
                'slot':     'answer',
                'value':    Variable('x')},
               {'action':   'pop_goal'}])             

R_larger_first_add_mul = Rule( # add or mul starting with larger operand
    name    = 'larger_first_add_mul',
    goal    = pat_solve_prob,
    conds   = [# problem is an unsolved problem
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               # first operand is smaller than second operand
               ('<',
                {'comparand1':  Variable('operand1'),
                 'comparand2':  Variable('operand2')}),
               # operation is addition or multiplication
               ('or',
                {'target':      Variable('operator'),
                 'pattern':     '+'},
                {'target':      Variable('operator'),
                 'pattern':     '*'})],
    acts    = make_SFA_acts(Variable('operator'), Variable('operand2'), Variable('operand1'), Variable('problem'), 'answer'))

R_sub_zero_from = Rule( # x minus zero is x
    name    = 'sub_zero_from',
    goal    = pat_solve_prob,
    conds   = [# problem is an unsolved subtraction problem
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operator'),
                'pattern':  '-'},
               # operand1 is a number and operand2 is zero
               {'target':     Variable('operand1'),
                'pattern':    Number()},
               {'target':     Variable('operand2'),
                'pattern':    Number(0)}],
    acts    = [# set problem answer to operand1 and pop goal
               {'action':   'set_feature',
                'chunk':    Variable('problem'),
                'slot':     'answer',
                'value':    Variable('operand1')},
               {'action':   'pop_goal'}])

R_mul_by_zero = Rule( # x times zero and zero times x are 0
    name    = 'mul_by_zero',
    goal    = pat_solve_prob,
    conds   = [# problem is an unsolved multiplication problem
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operator'),
                'pattern':  '*'},
               # one of the operands is zero
               ('or',
                {'target':     Variable('operand1'),
                 'pattern':    Number(0)},
                {'target':     Variable('operand2'),
                 'pattern':    Number(0)})],
    acts    = [# set problem answer to zero and pop goal
               {'action':   'set_feature',
                'chunk':    Variable('problem'),
                'slot':     'answer',
                'value':    Number(0)},
               {'action':   'pop_goal'}])

R_mul_by_one = Rule( # x times one is x
    name    = 'mul_by_one',
    goal    = pat_solve_prob,
    conds   = [# problem is an unsolved multiplication problem
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operator'),
                'pattern':  '*'},
               # one of the operands, x, is a number and the other is one
               ('or',
                ('and',
                 {'target':     Variable('operand1'),
                  'pattern':    Number(1)},
                 {'target':     Variable('operand2'),
                  'pattern':    Variable('x')},
                 {'target':     Variable('x'),
                  'pattern':    Number()}),
                ('and',
                 {'target':     Variable('operand2'),
                  'pattern':    Number(1)},
                 {'target':     Variable('operand1'),
                  'pattern':    Variable('x')},
                 {'target':     Variable('x'),
                  'pattern':    Number()}))],
    acts    = [# set problem answer to x and pop goal
               {'action':   'set_feature',
                'chunk':    Variable('problem'),
                'slot':     'answer',
                'value':    Variable('x')},
               {'action':   'pop_goal'}])

## Accumulator algorithm - a general procedure for accumulation, defined as using an action to increase a total iteratively until a counter reaches a target

R_acc_end = Rule( # If goal is accumulate and counter>=target, pop goal
    name    = 'acc_end',
    goal    = Goal('accumulate', {'accumulator': Variable('accumulator')}),
    conds   = [{'target':       Variable('accumulator'),
                'pattern':      pat_accumulator},
               ('>=',
                {'comparand1':  Variable('counter'),
                 'comparand2':  Variable('target')})],
    acts    = [{'action':       'pop_goal'}])

R_acc_once = Rule( # If goal is accumulate and counter<target, increment counter and create goal to do accumulator action
    name    = 'acc_once',
    goal    = Goal('accumulate', {'accumulator': Variable('accumulator')}),
    conds   = [{'target':       Variable('accumulator'),
                'pattern':      pat_accumulator},
               ('<',
                {'comparand1':  Variable('counter'),
                 'comparand2':  Variable('target')})],
    acts    = [{'action':       'increment',
                'chunk':        Variable('accumulator'),
                'slot':         'counter',
                'by':           1},
               {'action':       'push_goal',
                'goal':         Goal('do_accumulator_action', {'accumulator': Variable('accumulator')})}])

R_acc_skip = Rule( # If goal is accumulate and counter<target, increment counter only
    name    = 'acc_skip',
    goal    = R_acc_once.goal,
    conds   = R_acc_once.conds,
    acts    = [{'action':       'increment',
                'chunk':        Variable('accumulator'),
                'slot':         'counter',
                'by':           1}])

R_acc_extra = Rule( # If goal is accumulate and counter<target, create goal to do accumulator action only
    name    = 'acc_extra',
    goal    = R_acc_once.goal,
    conds   = R_acc_once.conds,
    acts    = [{'action':       'push_goal',
                'goal':         Goal('do_accumulator_action', {'accumulator': Variable('accumulator')})}])


## Addition and multiplication using accumulator algorithm

R_add_count = Rule( # If goal is solve unsolved whole add prob with operands in [1,10], and no fact retrieved, solve by counting
    name    = 'add_count',
    goal    = pat_solve_prob,
    conds   = [{'target':       Variable('problem'),
                'pattern':      pat_prob},
               {'target':       Variable('answer'),
                'pattern':      'BLANK'},
               {'target':       Variable('operator'),
                'pattern':      '+'},
               # no fact has been retrieved
               {'target':       Variable('retrieved_fact'),
                'pattern':      'BLANK'},
               # operand1 is a whole number in [1,10]
               {'target':       Variable('operand1'),
                'pattern':      pat_whole},
               ('<', {
                'comparand1':   Variable('operand1'),
                'comparand2':   Number(11)}),
               ('not',
                {'target':      Variable('operand1'),
                 'pattern':     Number(0)}),
               # operand2 is a whole number in [1,10]
               {'target':       Variable('operand2'),
                'pattern':      pat_whole},
               ('<', {
                'comparand1':   Variable('operand2'),
                'comparand2':   Number(11)}),
               ('not',
                {'target':      Variable('operand2'),
                 'pattern':     Number(0)})],
    acts    = [{'action':       'create_chunk',
                'variable':     'accumulator',
                'chunk_type':   Accumulator,
                'features':     {
                    'total':    Variable('operand1'),
                    'counter':  0,
                    'target':   Variable('operand2'),
                    'action':   'count',
                    'amount':   1},
                'copy':         True},
               {'action':   'push_goal',
                'goal':     Goal('deferred_action', {
                    'action':   'set_feature',
                    'features': {
                        'chunk':        Variable('problem'),
                        'slot':         'answer',
                        'from':         Variable('accumulator'),
                        'from_slot':    'total'}})},
               {'action':   'push_goal',
                'goal':     Goal('accumulate', {'accumulator': Variable('accumulator')})}])

R_mul_rep_add = Rule( # If goal is solve unsolved whole mul prob with operands in [2,10], and no fact retrieved, solve by repeated addition
    name    = 'mul_rep_add',
    goal    = pat_solve_prob,
    conds   = [{'target':       Variable('problem'),
                'pattern':      pat_prob},
               {'target':       Variable('answer'),
                'pattern':      'BLANK'},
               {'target':       Variable('operator'),
                'pattern':      '*'},
               # no fact has been retrieved
               {'target':       Variable('retrieved_fact'),
                'pattern':      'BLANK'},
               # operand1 is a whole number in [2,10]
               {'target':       Variable('operand1'),
                'pattern':      pat_whole},
               ('<', {
                'comparand1':   Number(1),
                'comparand2':   Variable('operand1')}),
               ('<', {
                'comparand1':   Variable('operand1'),
                'comparand2':   Number(11)}),
               # operand2 is a whole number in [1,10]
               {'target':       Variable('operand2'),
                'pattern':      pat_whole},
               ('<', {
                'comparand1':   Number(1),
                'comparand2':   Variable('operand2')}),
               ('<', {
                'comparand1':   Variable('operand2'),
                'comparand2':   Number(11)})],
    acts    = [{'action':       'create_chunk',
                'variable':     'accumulator',
                'chunk_type':   Accumulator,
                'features':     {
                    'total':    Variable('operand1'),
                    'counter':  1,
                    'target':   Variable('operand2'),
                    'action':   'add',
                    'amount':   Variable('operand1')},
                'copy':         True},
               {'action':   'push_goal',
                'goal':     Goal('deferred_action', {
                    'action':   'set_feature',
                    'features': {
                        'chunk':        Variable('problem'),
                        'slot':         'answer',
                        'from':         Variable('accumulator'),
                        'from_slot':    'total'}})},
               {'action':   'push_goal',
                'goal':     Goal('accumulate', {'accumulator': Variable('accumulator')})}])

R_acc_count = Rule( # If goal is do_accumulator_action and action is count, pop goal and increment running total by accumulator amount
    name    = 'acc_count',
    goal    = Goal('do_accumulator_action', {'accumulator': Variable('accumulator')}),
    conds   = [{'target':       Variable('accumulator'),
                'pattern':      pat_accumulator},
               {'target':       Variable('action'),
                'pattern':      'count'}],
    acts    = [{'action':       'pop_goal'},
               {'action':       'increment',
                'chunk':        Variable('accumulator'),
                'slot':         'total',
                'by':           Variable('amount')}])

R_acc_add = Rule( # If goal is do_accumulator_action and action is add, pop goal and create goals to add amount to running total
    name    = 'acc_add',
    goal    = Goal('do_accumulator_action', {'accumulator': Variable('accumulator')}),
    conds   = [{'target':       Variable('accumulator'),
                'pattern':      pat_accumulator},
               {'target':       Variable('action'),
                'pattern':      'add'}],
    acts    = [{'action':       'pop_goal'}] +
                make_SFA_acts('+', Variable('total'), Variable('amount'), Variable('accumulator'), 'total'))


## Single digit arithmetic using "calculator" (i.e., no calculation errors)

R_add_fact = Rule( # retrieve addition fact for SD whole numbers > 0
    name    = 'add_fact',
    goal    = pat_solve_prob,
    conds   = R_add_count.conds,
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem')}])

R_sub_fact = Rule( # retrieve subtraction fact for a - b with a<=20 and b<=10
    name    = 'sub_fact',
    goal    = pat_solve_prob,
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('operator'),
                'pattern':  '-'},
               {'target':   Variable('operand1'),
                'pattern':  pat_whole},
               ('<', 
                    {'comparand1':  Variable('operand1'),
                     'comparand2':  Number(21)}),
               {'target':   Variable('operand2'),
                'pattern':  pat_whole}, 
               ('<', 
                    {'comparand1':  Variable('operand2'),
                     'comparand2':  Number(11)})],
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem')}])

R_sub_LbS = Rule( # for a-b with a<=20 and b<=10, subtract smaller from larger
    name    = 'sub_LbS',
    goal    = pat_solve_prob,
    conds   = R_sub_fact.conds,
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem'),
                'lg_by_sm': True}])

R_mul_fact = Rule( # retrieve addition fact for SD whole numbers > 0
    name    = 'mul_fact',
    goal    = pat_solve_prob,
    conds   = R_mul_rep_add.conds,
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem')}])

R_div_calculator = Rule( # use calculator for integer division
    name    = 'div_calculator',
    goal    = pat_solve_prob,
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               {'target':   Variable('operator'),
                'pattern':  ':'},
               ('or',
                {'target':  Variable('operand1'),
                 'pattern': pat_whole},
                {'target':  Variable('operand1'),
                 'pattern': pat_negint}),
               ('or',
                {'target':  Variable('operand2'),
                 'pattern': pat_whole},
                {'target':  Variable('operand2'),
                 'pattern': pat_negint})],
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem')}])

R_div_drop_rem = Rule( # use calculator for division and drop remainder if any
    name    = 'div_drop_rem',
    goal    = pat_solve_prob,
    conds   = R_div_calculator.conds,
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem'),
                'as_int':   True}])

R_div_LbS = Rule( # use calculator for division and divide larger by smaller
    name    = 'div_LbS',
    goal    = pat_solve_prob,
    conds   = R_div_calculator.conds + [
               ('!=',
                {'comparand1':  Variable('operand1'),
                 'comparand2':  Number(0)}),
               ('!=',
                {'comparand1':  Variable('operand2'),
                 'comparand2':  Number(0)})],
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem'),
                'lg_by_sm': True}])

R_div_LbS_drop_rem = Rule( # use calculator for division and divide larger by smaller
    name    = 'div_LbS_drop_rem',
    goal    = pat_solve_prob,
    conds   = R_div_LbS.conds,
    acts    = [{'action':   'pop_goal'},
               {'action':   'use_calculator',
                'problem':  Variable('problem'),
                'lg_by_sm': True,
                'as_int':   True}])


## Aggregating a List of Numbers (by adding or multiplying them)

R_LA_start = Rule( # Start list aggregation algorithm
    name    = 'list_aggregation_start',
    goal    = Goal('aggregate_list', {
                'list':     Variable('list'),
                'operator': Variable('operator')}),
    conds   = [{'target':   Variable('list'),
                'pattern':  pat_lagg},
               ('not',
                {'target':  Variable('length'),
                 'pattern': 2})],
    acts    = [# pop goal
               {'action':   'pop_goal'},
               # attend to first item in list
               {'action':   'attend_list',
                'chunk':    Variable('list'),
                'pos':      'first'},
               # create do_list_aggregation goal
               {'action':   'push_goal',
                'goal':     Goal('do_list_aggregation', {
                    'list':     Variable('list'),
                    'operator': Variable('operator')})}])

R_LA_two = Rule( # Simplified list aggregation for two-element list
    name    = 'list_aggregation_two',
    goal    = R_LA_start.goal,
    conds   = R_LA_start.conds[:-1] + [
              {'target':  Variable('length'),
               'pattern': 2}],
    acts    = [{'action':   'pop_goal'}
              ] + make_SFA_acts(Variable('operator'), Variable('first'), Variable('second'), Variable('list'), 'list_ans'))

R_LA_set = Rule( # In list aggregation, set list_ans to current element                
    name    = 'list_aggregation_set',
    goal    = Goal('do_list_aggregation', {
                'list':         Variable('list'),
                'operator':     Variable('operator')}),
    conds   = [# current focus is in list
               {'target':       Variable('list'),
                'pattern':      pat_lagg},
               {'target':       Variable('attn_in'),
                'pattern':      True},
               # current focus is not blank
               ('not',
                {'target':      Variable('current'),
                 'pattern':     'BLANK'}),
               # list_ans is blank
               {'target':       Variable('list_ans'),
                'pattern':      'BLANK'}],
    acts    = [# set list_ans to current element
               {'action':       'set_feature',
                'chunk':        Variable('list'),
                'slot':         'list_ans',
                'value':        Variable('current')},
               # shift focus to next list element
               {'action':       'shift_list',
                'chunk':        Variable('list'),
                'dir':          'next'}])

R_LA_calc = Rule( # In list aggregation, do next calculation
    name    = 'list_aggregation_calc',
    goal    = R_LA_set.goal,
    conds   = R_LA_set.conds[:-1] + [
               # same conds as R_LA_set, except list_ans is not blank
               ('not',
                {'target':      Variable('list_ans'),
                 'pattern':     'BLANK'})],
    acts    = [# create goal to shift focus to next list element
               {'action':       'push_goal',
                'goal':     Goal('deferred_action', {
                    'action':   'shift_list',
                    'features': {
                        'chunk':        Variable('list'),
                        'dir':          'next'}})}
               # aggregate current focus with list_ans
               ] + make_SFA_acts(Variable('operator'), Variable('list_ans'), Variable('current'), Variable('list'), 'list_ans'))

R_LA_skip = Rule( # In list aggregation, if current element is blank and you are adding, skip it
    name    = 'list_aggregation_skip',
    goal    = R_LA_set.goal,
    conds   = [# current focus is in list
               {'target':       Variable('list'),
                'pattern':      pat_lagg},
               {'target':       Variable('attn_in'),
                'pattern':      True},
               # current focus is blank
               {'target':       Variable('current'),
                'pattern':      'BLANK'},
               # operation is addition
               {'target':   Variable('operator'),
                'pattern':  '+'}],
    acts    = [# shift focus to next list element
               {'action':   'shift_list',
                'chunk':        Variable('list'),
                'dir':          'next'}])
                
R_LA_zero = Rule( # In list aggregation, if current element is blank and you are multiplying, set answer to zero and end
    name    = 'list_aggregation_zero',
    goal    = R_LA_set.goal,
    conds   = R_LA_skip.conds[:-1] + [
               # operation is multiplication
               {'target':   Variable('operator'),
                'pattern':  '*'}],
    acts    = [{'action':   'pop_goal'},
               {'action':       'set_feature',
                'chunk':        Variable('list'),
                'slot':         'list_ans',
                'value':        Number(0)}])
                    
R_LA_end = Rule( # End list aggregation algorithm
    name    = 'list_aggregation_end',
    goal    = R_LA_set.goal,
    conds   = [# focus is out of list
               {'target':   Variable('list'),
                'pattern':  pat_lagg},
               {'target':   Variable('attn_in'),
                'pattern':  False}],
    acts    = [{'action':   'pop_goal'}])


## MD expert rules

R_H2V_WN = Rule( # Solve MD whole number prob using vertical format
    name    = 'H2V_WN',
    goal    = pat_solve_prob,
    conds   = [# problem is a regular problem (not vertical) and unsolved
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               # both operands are whole numbers and at least one is >10
               {'target':   Variable('operand1'), 
                'pattern':  pat_whole},
               {'target':   Variable('operand2'), 
                'pattern':  pat_whole},
               ('or',
                ('<', {'comparand1': Number(10), 'comparand2': Variable('operand1')}),
                ('<', {'comparand1': Number(10), 'comparand2': Variable('operand2')})),
               # operation is addition, multiplication, or subtraction
               ('or',
                {'target':  Variable('operator'), 'pattern': '+'},
                {'target':  Variable('operator'), 'pattern': '-'},
                {'target':  Variable('operator'), 'pattern': '*'}),
               # if operation is subtraction, operand1 >= 20 or operand2 is multidigit
               ('or',
                ('not', {'target': Variable('operator'), 'pattern': '-'}),
                ('>=', {'comparand1': Variable('operand1'), 'comparand2': Number(20)}),
                {'target': Variable('operand2'), 'pattern': pat_MD_whole})],
    acts    = [ # create blank vproblem
               {'action':       'create_chunk',
                'variable':     'vproblem',
                'chunk_type':   vProblem,
                'features':     {
                    'operator':     Variable('operator'),
                    'operand1':     Variable('operand1'),
                    'operand2':     Variable('operand2')}},
               # create goal to set prob ans to vprob ans
               {'action':       'push_goal', 
                'goal':     Goal('deferred_action', {
                    'action':   'set_feature',
                    'features': {
                        'chunk':        Variable('problem'),
                        'slot':         'answer',
                        'from':         Variable('vproblem'),
                        'from_slot':    'answer'}})},
               # create goal to solve vprob
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {
                    'problem':      Variable('vproblem')})},
               # create goal to align operands in vprob
               {'action':       'push_goal',
                'goal':         Goal('align_operands', {
                    'vproblem':     Variable('vproblem'),
                    'align':        'right'})}])

R_align_right = Rule( # Align decimal operands at rightmost digit
    name    = 'align_right',
    goal    = Goal('align_operands', {
                'vproblem': Variable('vproblem'),
                'align':    'right'}),
    conds   = [{'target':   Variable('vproblem'),
                'pattern':  pat_vprob}],
    acts    = [{'action':   'pop_goal'},
               {'action':   'align_operands',
                'chunk':    Variable('operands'),
                'align':    'right'}])

R_VA_start = Rule( # Start vertical arithmetic
    name    = 'VA_start',
    goal    = Goal('solve_problem', {'problem':  Variable('vproblem')}),
    conds   = [# problem is an unsolved vproblem
               {'target':   Variable('vproblem'),
                'pattern':  pat_vprob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'}],
    acts    = [# create goal to start vertical arithmetic with state = CHOOSE_ALGORITHM
               {'action':   'push_goal',
                'goal':     Goal('do_valg', {
                    'algorithm':    None,
                    'vproblem':     Variable('vproblem'),
                    'state':        'CHOOSE_ALGORITHM'})}])
                    
# Algorithm for vertical (column) addition (or subtraction): 'VAS'

R_choose_VAS_AS = Rule( # Choose VAS algorithm for vertical addition or subtraction
    name    = 'choose_VAS_AS',
    goal    = pat_do_valg,
    conds   = [# state is CHOOSE_ALGORITHM
               {'target':       Variable('state'),
                'pattern':      'CHOOSE_ALGORITHM'},
               # operation is addition or subtraction
               {'target':   Variable('vproblem'),
                'pattern':  pat_vprob},
               ('or',
                {'target':  Variable('operator'),
                 'pattern': '+'},
                {'target':  Variable('operator'),
                 'pattern': '-'})],
    acts    = [# set algorithm state to NEXT_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'NEXT_CALC'},
               # set algorithm to VAS
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'algorithm',
                'value':        'VAS'},
               # attend to rightmost column
               {'action':   'attend_col',
                'chunk':    Variable('vproblem'),
                'pos':      'last'}])

R_VA_next_calc = Rule( # In VAS state NEXT_CALC, for operation other than subtraction, do next calculation
    name    = 'VA_next_calc',
    goal    = pat_do_valg,
    conds   = [# algorithm is VAS
               {'target':       Variable('algorithm'),
                'pattern':      'VAS'},
               # state is NEXT_CALC
               {'target':       Variable('state'),
                'pattern':      'NEXT_CALC'},
               # operation is not subtraction
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               ('not',
                {'target':      Variable('operator'),
                 'pattern':     '-'}),
               # current column is not blank
               ('not',
                {'target':      Variable('curr_col'),
                 'pattern':     'BLANK'})],
    acts    = [# set algorithm state to END_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'END_CALC'},
               # create goal to aggregate current column
               {'action':       'push_goal',
                'goal':         Goal('aggregate_list', {
                    'list':         Variable('curr_col'),
                    'operator':     Variable('operator')})}])

R_VA_lone_carry = Rule( # In VAS state NEXT_CALC, for addition, handle lonely carry
    name    = 'VA_lone_carry',
    goal    = pat_do_valg,
    conds   = R_VA_next_calc.conds[:-1] + [
               # same conditions as R_VA_next_calc, except:
               # in vproblem, current column is blank and carry is not blank
               {'target':       Variable('curr_col'),
                'pattern':      'BLANK'},
               ('not',
                {'target':      Variable('this_carry'),
                 'pattern':     'BLANK'})],
    acts    = [{'action':       'prepend_digits',
                'number':       Variable('answer'),
                'digits':       Variable('this_carry')},
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'this_carry',
                'value':        None}])                      

R_VS_next_calc = Rule( # In VAS state NEXT_CALC, for subtraction, do next calculation
    name    = 'VS_next_calc',
    goal    = pat_do_valg,
    conds   = [# algorithm is VAS
               {'target':       Variable('algorithm'),
                'pattern':      'VAS'},
               # state is NEXT_CALC
               {'target':       Variable('state'),
                'pattern':      'NEXT_CALC'},
               # operation is subtraction
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('operator'),
                'pattern':      '-'},
               # current column is not blank
               ('not',
                {'target':      Variable('curr_col'),
                 'pattern':     'BLANK'}),
               # first number in curr_col >= second number in curr_col
               {'target':       Variable('curr_col'),
                'pattern':      pat_lagg},
               ('>=',
                {'comparand1':   Variable('first'),
                 'comparand2':   Variable('second')})],
    acts    = [# set algorithm state to END_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'END_CALC'},
               # create goal to aggregate current column
               {'action':       'push_goal',
                'goal':         Goal('aggregate_list', {
                    'list':         Variable('curr_col'),
                    'operator':     Variable('operator')})}])

R_VS_next_calc_swap = Rule( # In VAS state NEXT_CALC, for subtraction, do next calculation but swap operands
    name    = 'VS_next_calc_swap',
    goal    = pat_do_valg,
    conds   = [# algorithm is VAS
               {'target':       Variable('algorithm'),
                'pattern':      'VAS'},
               # state is NEXT_CALC
               {'target':       Variable('state'),
                'pattern':      'NEXT_CALC'},
               # operation is subtraction
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('operator'),
                'pattern':      '-'},
               # current column is not blank
               ('not',
                {'target':      Variable('curr_col'),
                 'pattern':     'BLANK'}),
               # first number in curr_col < second number in curr_col
               {'target':       Variable('curr_col'),
                'pattern':      pat_lagg},
               ('<',
                {'comparand1':   Variable('first'),
                 'comparand2':   Variable('second')})],
    acts    = [# set algorithm state to END_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'END_CALC'}] +
               # create goal to subtract current column in reverse order
               make_SFA_acts('-', Variable('second'), Variable('first'), Variable('curr_col'), 'list_ans'))

R_VS_borrow_start = Rule( # In VAS state NEXT_CALC, for subtraction, start borrowing
    name    = 'VS_borrow_start',
    goal    = pat_do_valg,
    conds   = R_VS_next_calc.conds[:-1] + [
               # same conditions as R_VS_next_calc, except:
               # first number in curr_col < second number in curr_col
               {'target':       Variable('first'),
                'pattern':      Number()},
               ('<',
                {'comparand1':   Variable('first'),
                 'comparand2':   Variable('second')})],
    acts    = [# create goal to borrow from previous column to this column
               {'action':       'push_goal',
                'goal':         Goal('borrow_to', {
                    'vproblem':     Variable('vproblem')})}])

R_VS_borrow_to = Rule( # If goal is borrow into current column, create goals to do so
    name    = 'VS_borrow_to',
    goal    = Goal('borrow_to', {
                'vproblem':     Variable('vproblem')}),
    conds   = [{'target':       Variable('vproblem'),
                'pattern':      pat_vprob}],
    acts    = [# pop goal
               {'action':       'pop_goal'},
               # create goal to prepend 1 to first digit of curr_col
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'prepend_digits',
                    'features':     {
                        'chunk':        Variable('curr_col'),
                        'slot':         'first',
                        'digits':       Number(1)}})},
               # shift attention left
               {'action':       'shift_col',
                'chunk':        Variable('vproblem'),
                'dir':          'prev'},
               # create borrow_from goal
               {'action':       'push_goal',
                'goal':         Goal('borrow_from', {
                    'vproblem':     Variable('vproblem')})}])
                    
R_VS_borrow_from_nonzero = Rule( # If goal is borrow_from and curr_col first not zero and not None, do the borrow
    name    = 'VS_borrow_from_nonzero',
    goal    = Goal('borrow_from', {
                'vproblem':     Variable('vproblem')}),
    conds   = [{'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('curr_col'),
                'pattern':      pat_lagg},
               ('not',
                {'target':      Variable('first'),
                 'pattern':     Number(0)}),
               ('not',
                {'target':      Variable('first'),
                 'pattern':     'BLANK'})],
    acts    = [# pop goal
               {'action':       'pop_goal'},
               # decrement first digit of current column
               {'action':       'decrement',
                'chunk':        Variable('curr_col'),
                'slot':         'first'},
               # shift attention right
               {'action':       'shift_col',
                'chunk':        Variable('vproblem'),
                'dir':          'next'}])

R_VS_borrow_from_zero = Rule( # If goal is borrow_from and curr_col first is zero, borrow to curr_col
    name    = 'VS_borrow_from_zero',
    goal    = R_VS_borrow_from_nonzero.goal,
    conds   = R_VS_borrow_from_nonzero.conds[:-2] + [
               {'target':      Variable('first'),
                'pattern':     Number(0)}],
    acts    = [{'action':       'push_goal',
                'goal':         Goal('borrow_to', {
                    'vproblem':     Variable('vproblem')})}])
                    
R_VS_borrow_from_fail = Rule( # If goal is borrow_from and curr_col first zero or None, skip the borrow
    name    = 'VS_borrow_from_fail',
    goal    = Goal('borrow_from', {
                'vproblem':     Variable('vproblem')}),
    conds   = [{'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               ('or',
                {'target':      Variable('curr_col'),
                 'pattern':     'BLANK'},
                ('and',
                 {'target':     Variable('curr_col'),
                  'pattern':    pat_lagg},
                 ('or',
                  {'target':    Variable('first'),
                   'pattern':   'BLANK'},
                  {'target':    Variable('first'),
                   'pattern':   Number(0)})))],
    acts    = [# pop goal
               {'action':       'pop_goal'},
               # shift attention right
               {'action':       'shift_col',
                'chunk':        Variable('vproblem'),
                'dir':          'next'}])

R_VAS_finish = Rule( # In VAS state NEXT_CALC, finish the algorithm
    name    = 'VAS_finish',
    goal    = pat_do_valg,
    conds   = [# algorithm is VAS
               {'target':       Variable('algorithm'),
                'pattern':      'VAS'},
               # state is NEXT_CALC
               {'target':       Variable('state'),
                'pattern':      'NEXT_CALC'},
               # in vproblem, current column is blank and carry is blank
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('curr_col'),
                'pattern':      'BLANK'},
               {'target':       Variable('this_carry'),
                'pattern':      'BLANK'}],
    acts    = [{'action':       'pop_goal'}])

R_VAS_end_calc = Rule( # In VAS state END_CALC, end current calculation
    name    = 'VAS_end_calc',
    goal    = pat_do_valg,
    conds   = [# algorithm is VAS
               {'target':       Variable('algorithm'),
                'pattern':      'VAS'},
               # state is END_CALC
               {'target':       Variable('state'),
                'pattern':      'END_CALC'},
               # extract curr_col from vproblem for use below
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('curr_col'),
                'pattern':      pat_lagg},
               # in vproblem, this_carry is blank and list_ans has only one digit
               {'target':       Variable('this_carry'),
                'pattern':     'BLANK'},
               {'target':       Variable('list_ans'),
                'pattern':      Number(features={'multidig': False})}],
    acts    = [# set algorithm state to SHIFT_ATTN
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'SHIFT_ATTN'},
               # prepend sum of current column to final answer
               {'action':       'prepend_digits',
                'chunk':        Variable('vproblem'),
                'slot':         'answer',
                'digits':       Variable('list_ans')}])

R_VAS_do_carry = Rule( # In VAS state END_CALC, extract carry from column answer
    name    = 'VAS_do_carry',
    goal    = pat_do_valg,
    conds   = R_VAS_end_calc.conds[:-1] + [
               # same conditions as for VAS_end_calc, except:
               # in vproblem, in curr_col, list_ans has multiple digits
               {'target':       Variable('list_ans'),
                'pattern':      Number(features={'multidig': True})}],
    acts    = [# split column sum into left and right parts
               {'action':       'decompose_whole',
                'decomp_type':  'right_digit',
                'source':       Variable('list_ans'),
                'variables':    ['ldig', 'rdig']},
               # set column sum to right part
               {'action':       'set_feature',
                'chunk':        Variable('curr_col'),
                'slot':         'list_ans',
                'value':        Variable('rdig')},
               # add carry to matrix left of current position
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'next_carry',
                'value':        Variable('ldig')}])
                
R_VAS_add_carry = Rule( # In VAS state END_CALC, add carry to current calculation
    name    = 'VAS_add_carry',
    goal    = pat_do_valg,
    conds   = R_VAS_end_calc.conds[:-2] + [
               # same conditions as R_VAS_end_calc, except:
               # in vproblem, this_carry is not blank and no requirement for list_ans
               ('not',
                {'target':      Variable('this_carry'),
                 'pattern':     'BLANK'})],
    acts    = [# set this_carry to none
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'this_carry',
                'value':        None}
               # add carry to current column sum
               ] + make_SFA_acts('+', Variable('list_ans'), Variable('this_carry'), Variable('curr_col'), 'list_ans'))

R_VAS_shift_attn = Rule( # In VAS state SHIFT_ATTN, shift attention left
    name    = 'VAS_shift_attn',
    goal    = pat_do_valg,
    conds   = [# algorithm is VAS
               {'target':       Variable('algorithm'),
                'pattern':      'VAS'},
               # state is SHIFT_ATTN
               {'target':       Variable('state'),
                'pattern':      'SHIFT_ATTN'}],
    acts    = [# set algorithm state to NEXT_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'NEXT_CALC'},
               # shift attention left
               {'action':       'shift_col',
                'chunk':        Variable('vproblem'),
                'dir':          'prev'},
               # in vproblem, turn "next carry" into "this carry"
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'this_carry',
                'from':         Variable('vproblem'),
                'from_slot':    'next_carry'},
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'next_carry',
                'value':        None}])

# Algorithm for vertical format multiplication with SD operand2: 'VM'

R_choose_VM_M = Rule( # Choose VM algorithm for vertical multiplication
    name    = 'choose_VM_M',
    goal    = R_choose_VAS_AS.goal,
    conds   = R_choose_VAS_AS.conds[:-1] + [
               # same conditions as R_choose_VAS_AS except operation is multiplication
               {'target':   Variable('operator'),
                'pattern':  '*'}],
    acts    = [# set algorithm state to NEXT_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'NEXT_CALC'},
               # set algorithm to VM
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'algorithm',
                'value':        'VM'},
               # attend to rightmost digit of each operand
               {'action':   'attend_dig',
                'chunk':    Variable('operand1'),
                'pos':      'last'},
               {'action':   'attend_dig',
                'chunk':    Variable('operand2'),
                'pos':      'last'}])

R_VM_next_calc = Rule( # In VM state NEXT_CALC, do next calculation
    name    = 'VM_next_calc',
    goal    = pat_do_valg,
    conds   = [# algorithm is VM
               {'target':       Variable('algorithm'),
                'pattern':      'VM'},
               # state is NEXT_CALC
               {'target':       Variable('state'),
                'pattern':      'NEXT_CALC'},
               # extract variables from vproblem for use below
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('operand1'),
                'pattern':      Number(features={'current': Variable('op1dig')})},
               {'target':       Variable('operand2'),
                'pattern':      Number(features={'current': Variable('op2dig')})},
               # in vproblem, current digits are not blank
               ('not',
                {'target':      Variable('op1dig'),
                 'pattern':     'BLANK'}),
               ('not',
                {'target':      Variable('op2dig'),
                 'pattern':     'BLANK'})],
    acts    = [# set algorithm state to END_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'END_CALC'}] +
               # calculate product of current digits, and store result in curr_ans (in vprob)
              make_SFA_acts('*', Variable('op1dig'), Variable('op2dig'), Variable('vproblem'), 'curr_ans'))

R_VM_lone_carry = Rule( # In VM state NEXT_CALC, handle lonely carry
    name    = 'VM_lone_carry',
    goal    = pat_do_valg,
    conds   = R_VM_next_calc.conds[:-2] + [
               # same conditions as R_VM_next_calc, except:
               # in vproblem, current digit in operand1 is blank (operand2 doesn't matter) and carry is not blank
               {'target':  Variable('op1dig'),
                'pattern': 'BLANK'},
               ('not',
                {'target':      Variable('this_carry'),
                 'pattern':     'BLANK'})],
    acts    = R_VA_lone_carry.acts)

R_VM_finish_sd = Rule( # In VM state NEXT_CALC with SD operand2, finish
    name    = 'VM_finish_sd',
    goal    = pat_do_valg,
    conds   = R_VM_next_calc.conds[:-2] + [
               # same conditions as R_VM_next_calc, except:
               # in vproblem, current digit in operand1 is blank, operand2 is single digit, and carry is blank
               {'target':   Variable('operand1'),
                'pattern':  Number(features={'current': Variable('op1dig')})},
               {'target':  Variable('op1dig'),
                'pattern': 'BLANK'},
               # operand2 is a single digit number
               {'target':   Variable('operand2'),
                'pattern':  Number(features={
                    'current':  Variable('op2dig'),
                    'multidig': False})},
               # carry is blank
               {'target':   Variable('this_carry'),
                'pattern':  'BLANK'}],
    acts    = [{'action':       'pop_goal'}])

R_VM_end_calc = Rule( # In VM state END_CALC, end current calculation
    name    = 'VM_end_calc',
    goal    = pat_do_valg,
    conds   = [# algorithm is VM
               {'target':       Variable('algorithm'),
                'pattern':      'VM'},
               # state is END_CALC
               {'target':       Variable('state'),
                'pattern':      'END_CALC'},
               # extract variables for use below
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               # in vproblem, this_carry is blank and curr_ans has only one digit
               {'target':       Variable('this_carry'),
                'pattern':      'BLANK'},
               {'target':       Variable('curr_ans'),
                'pattern':      Number(features={'multidig': False})}],
    acts    = [# set algorithm state to SHIFT_ATTN
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'SHIFT_ATTN_1'},
               # prepend sum of current column to final answer
               {'action':       'prepend_digits',
                'chunk':        Variable('vproblem'),
                'slot':         'answer',
                'digits':       Variable('curr_ans')}])

R_VM_do_carry = Rule( # In VM state END_CALC, extract carry from current calc ans
    name    = 'VM_do_carry',
    goal    = pat_do_valg,
    conds   = R_VM_end_calc.conds[:-1] + [
               # same conditions as for VM_end_calc, except:
               # in vproblem, in curr_ans has multiple digits
               {'target':       Variable('curr_ans'),
                'pattern':      Number(features={'multidig': True})}],
    acts    = [# split curr_ans into left and right parts
               {'action':       'decompose_whole',
                'decomp_type':  'right_digit',
                'source':       Variable('curr_ans'),
                'variables':    ['ldig', 'rdig']},
               # set curr_ans to right part of curr_ans
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'curr_ans',
                'value':        Variable('rdig')},
               # set next_carry to left part of curr_ans
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'next_carry',
                'value':        Variable('ldig')}])

R_VM_add_carry = Rule( # In VM state END_CALC, add carry to current calculation answer
    name    = 'VM_add_carry',
    goal    = pat_do_valg,
    conds   = R_VM_end_calc.conds[:-2] + [
               # same conditions as R_VM_end_calc, except:
               # in vproblem, this_carry is not blank and no requirement for curr_ans
               ('not',
                {'target':      Variable('this_carry'),
                 'pattern':     'BLANK'})],
    acts    = [# set this_carry to none
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'this_carry',
                'value':        None}
               # add carry to current column sum
               ] + make_SFA_acts('+', Variable('curr_ans'), Variable('this_carry'), Variable('vproblem'), 'curr_ans'))

R_VM_shift1 = Rule( # In VM state SHIFT_ATTN_1, shift attention left
    name    = 'VM_shift1',
    goal    = pat_do_valg,
    conds   = [# algorithm is VM
               {'target':       Variable('algorithm'),
                'pattern':      'VM'},
               # state is SHIFT_ATTN_1
               {'target':       Variable('state'),
                'pattern':      'SHIFT_ATTN_1'},
               # extract vproblem features for use below
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob}],
    acts    = [# set algorithm state to NEXT_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'NEXT_CALC'},
               # shift attention left in operand 1
               {'action':       'shift_dig',
                'chunk':        Variable('operand1'),
                'dir':          'prev'},
               # turn "next carry" into "this carry"
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'this_carry',
                'value':        Variable('next_carry')},
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'next_carry',
                'value':        None}])

# Additional rules for vertical format multiplication with MD operand2

R_VM_shift2 = Rule( # In VM state NEXT_CALC, reset in operand1 and shift in operand2
    name    = 'VM_shift2',
    goal    = pat_do_valg,
    conds   = R_VM_next_calc.conds[:-2] + [
               # same conditions as R_VM_next_calc, except:
               # current digit in operand1 is blank
               {'target':       Variable('op1dig'),
                'pattern':      'BLANK'},
               # carry is blank
               {'target':       Variable('this_carry'),
                'pattern':      'BLANK'},
               # currently attended digit in operand2 is nonempty
               ('not',
                {'target':      Variable('op2dig'),
                 'pattern':     'BLANK'}),
               # operand2 is a multidigit number
               {'target':       Variable('operand2'),
                'pattern':      Number(features={
                    'multidig':     True})}],
    acts    = [# set algorithm state to NEW_ROW
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'NEW_ROW'},
               # reset attention in operand1 and shift left in operand2
               {'action':       'attend_dig',
                'chunk':        Variable('operand1'),
                'pos':          'last'},
               {'action':       'shift_dig',
                'chunk':        Variable('operand2'),
                'dir':          'prev'},
               # move answer to partial product list
               {'action':       'append_arithmat',
                'chunk':        Variable('vproblem'),
                'slot':         'part_prods',
                'element':      Variable('answer')},
               {'action':       'set_feature',
                'chunk':        Variable('vproblem'),
                'slot':         'answer',
                'value':        None},
               # increment to-be-prepended zeroes 
               {'action':       'prepend_digits',
                'chunk':        Variable('vproblem'),
                'slot':         'prep_zeros',
                'digits':       0}])

R_VM_shift2_no_zeros = Rule( # VM_shift2 but don't increment to-be-prepended zeroes
    name    = 'VM_shift2_no_zeros',
    goal    = R_VM_shift2.goal,
    conds   = R_VM_shift2.conds,
    acts    = R_VM_shift2.acts[:-1])

R_VM_new_row = Rule( # In VM state NEW_ROW, start next partial product
    name    = 'VM_new_row',
    goal    = pat_do_valg,
    conds   = [# algorithm is VM
               {'target':       Variable('algorithm'),
                'pattern':      'VM'},
               # state is NEW_ROW
               {'target':       Variable('state'),
                'pattern':      'NEW_ROW'},
               # extract variables from vproblem for use below
               {'target':       Variable('vproblem'),
                'pattern':      pat_vprob},
               {'target':       Variable('operand1'),
                'pattern':      Number(features={'current': Variable('op1dig')})},
               {'target':       Variable('operand2'),
                'pattern':      Number(features={'current': Variable('op2dig')})},
               # attended digit in operand2 is not blank
               ('not',
                {'target':  Variable('op2dig'),
                 'pattern': 'BLANK'})],
    acts    = [# set algorithm state to NEXT_CALC
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'state',
                'value':        'NEXT_CALC'},
               # prepend zeroes to current partial product
               {'action':       'prepend_digits',
                'chunk':        Variable('vproblem'),
                'slot':         'answer',
                'digits':       Variable('prep_zeros')}])

R_VM_add_parts = Rule( # While doing vertical multiplication, add partial products
    name    = 'VM_add_parts',
    goal    = pat_do_valg,
    conds   = R_VM_new_row.conds[:-1] + [
               # same conditions as R_VM_new_row, except:
               # attended digit in operand2 is blank
               {'target':       Variable('op2dig'),
                'pattern':      'BLANK'}],
    acts    = [# pop goal
               {'action':       'pop_goal'},
               # create problem representing sum of partial products
               {'action':       'create_chunk',
                'variable':     'part_prod_sum',
                'chunk_type':   vProblem,
                'features':     {
                    'operator':     '+',
                    'operands':     Variable('part_prods')}},
               # create goal to set orig vprob ans to partial product sum
               {'action':       'push_goal', 
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features':     {
                        'chunk':        Variable('vproblem'),
                        'slot':         'answer',
                        'from':         Variable('part_prod_sum'),
                        'from_slot':    'answer'}})},
               # attend to rightmost column
               {'action':   'attend_col',
                'chunk':    Variable('part_prod_sum'),
                'pos':      'last'},
               # create goal to do VAS algorithm with state = NEXT_CALC
               {'action':   'push_goal',
                'goal':     Goal('do_valg', {
                    'algorithm':    'VAS',
                    'vproblem':     Variable('part_prod_sum'),
                    'state':        'NEXT_CALC'})}])


## Decimals

# ADBD: align decimals (of operands), bring decimal (from operands into answer)

R_ADBD_AS = Rule( # ADBD for add or sub
    name    = 'ADBD_AS',
    goal    = pat_solve_prob,
    conds   = [# problem is a regular problem (not vertical) and unsolved
               {'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               # both operands have digits and at least one is a decimal
               {'target':   Variable('operand1'),
                'pattern':  Number(features={
                    'subtype':  Variable('op1type'),
                    'digits':   Variable('op1dig')})},
               {'target':   Variable('operand2'),
                'pattern':  Number(features={
                    'subtype':  Variable('op2type'),
                    'digits':   Variable('op2dig')})},
               ('or',
                {'target':  Variable('op1type'),
                 'pattern': 'decimal'},
                {'target':  Variable('op2type'),
                 'pattern': 'decimal'}),
               # operation is addition, multiplication, or subtraction
               ('or',
                {'target':  Variable('operator'), 'pattern': '+'},
                {'target':  Variable('operator'), 'pattern': '-'})],
    acts    = [# create blank vproblem
               {'action':       'create_chunk',
                'variable':     'vproblem',
                'chunk_type':   vProblem,
                'features':     {
                    'operator':     Variable('operator'),
                    'operand1':     Variable('operand1'),
                    'operand2':     Variable('operand2')}},
               # create goal to set prob ans to vprob ans
               {'action':       'push_goal', 
                'goal':     Goal('deferred_action', {
                    'action':   'set_feature',
                    'features': {
                        'chunk':        Variable('problem'),
                        'slot':         'answer',
                        'from':         Variable('vproblem'),
                        'from_slot':    'answer'}})},
               # create goal to bring decimal from operands to answer
               {'action':       'push_goal',
                'goal':         Goal('place_decimal', {
                    'vproblem':     Variable('vproblem'),
                    'method':       'bring_decimal'})},
               # create goal to solve vproblem
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {
                    'problem':      Variable('vproblem')})},
               # create goal to align operands at decimal point
               {'action':   'push_goal',
                'goal':     Goal('align_operands', features={
                    'vproblem':     Variable('vproblem'),
                    'align':        'decimal'})}])

R_ADBD_OG = Rule( # ADBD for add, sub, or mul
    name    = 'ADBD_OG',
    goal    = pat_solve_prob,
    conds   = R_ADBD_AS.conds[:-1] + 
              [# any operation but division is OK
               ('not',
                {'target':  Variable('operator'), 'pattern': ':'})],
    acts    = R_ADBD_AS.acts)

# ARAD: align rightmost (digits of operands), add decimal (digits of operands to determine position of decimal in answer)

R_ARAD_M = Rule( # ARAD for mul
    name    = 'ARAD_M',
    goal    = pat_solve_prob,
    conds   = R_ADBD_AS.conds[:-1] + [
               # operation is addition, multiplication, or subtraction
               {'target':  Variable('operator'), 'pattern': '*'}],
    acts    = # create blank vproblem and goal to set prob ans to vprob ans
              R_ADBD_AS.acts[0:2] + 
              [# create goal to add decimal digits of operands to determine position of decimal in answer
               {'action':       'push_goal',
                'goal':         Goal('place_decimal', {
                    'vproblem':     Variable('vproblem'),
                    'method':       'add_dd'})},
               # create goal to solve vproblem
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {
                    'problem':      Variable('vproblem')})},
               # create goal to align operands at decimal point
               {'action':   'push_goal',
                'goal':     Goal('align_operands', features={
                    'vproblem':     Variable('vproblem'),
                    'align':        'right'})}])

R_ARAD_OG = Rule( # ARAD for add, sub, or mul
    name    = 'ARAD_OG',
    goal    = pat_solve_prob,
    conds   = R_ARAD_M.conds[:-1] + 
              [# any operation but division is OK
               ('not',
                {'target':  Variable('operator'), 'pattern': ':'})],
    acts    = R_ARAD_M.acts)

# Execution rules

R_align_dec_no_zeros = Rule( # Align decimal operands at decimal point without appending zeros
    name    = 'align_dec_no_zeros',
    goal    = Goal('align_operands', {
                'vproblem': Variable('vproblem'),
                'align':    'decimal'}),
    conds   = [{'target':   Variable('vproblem'),
                'pattern':  pat_vprob}],
    acts    = [{'action':   'pop_goal'},
               {'action':   'align_operands',
                'chunk':    Variable('vproblem'),
                'align':    'decimal'}])
                
R_align_dec_app_zeros = Rule( # Align decimal operands at decimal point with appending zeros
    name    = 'align_dec_app_zeros',
    goal    = R_align_dec_no_zeros.goal,
    conds   = [{'target':   Variable('vproblem'),
                'pattern':  pat_vprob},
               {'target':   Variable('operand1'),
                'pattern':  Number(features={'num_dec_dig': Variable('op1ndd')})},
               {'target':   Variable('operand2'),
                'pattern':  Number(features={'num_dec_dig': Variable('op2ndd')})},
               ('not',
                {'target':  Variable('op1ndd'),
                'pattern':  Variable('op2ndd')})],
    acts    = [{'action':   'pop_goal'},
               {'action':   'align_operands',
                'chunk':    Variable('vproblem'),
                'align':    'decimal',
                'append':   True}])

R_align_right = Rule( # Align decimal operands at rightmost digit
    name    = 'align_right',
    goal    = Goal('align_operands', {
                'vproblem': Variable('vproblem'),
                'align':    'right'}),
    conds   = [{'target':   Variable('vproblem'),
                'pattern':  pat_vprob}],
    acts    = [{'action':   'pop_goal'},
               {'action':   'align_operands',
                'chunk':    Variable('operands'),
                'align':    'right'}])

R_bring_dec = Rule( # Bring decimal from operands into answer
    name    = 'bring_dec',
    goal    = Goal('place_decimal', {
                'vproblem': Variable('vproblem'),
                'method':   'bring_decimal'}),
    conds   = [{'target':   Variable('vproblem'),
                'pattern':  pat_vprob}],
    acts    = [{'action':   'pop_goal'},
               {'action':   'get_dec_dig_from_op',
                'vproblem': Variable('vproblem'),
                'variable': 'dd'},
               {'action':   'set_dec_dig_in_ans',
                'vproblem': Variable('vproblem'),
                'dd':       Variable('dd')}])

R_add_dd = Rule( # Add decimal digits of operands to determine position of decimal in answer
    name    = 'add_dd',
    goal    = Goal('place_decimal', {
                'vproblem': Variable('vproblem'),
                'method':   'add_dd'}),
    conds   = [{'target':   Variable('vproblem'),
                'pattern':  pat_vprob},
               {'target':   Variable('operand1'),
                'pattern':  Number(features={
                    'num_dec_dig':  Variable('op1ndd')})},
               {'target':   Variable('operand2'),
                'pattern':  Number(features={
                    'num_dec_dig':  Variable('op2ndd')})}],
    acts    = [{'action':       'pop_goal'},
               {'action':       'create_chunk',
                'variable':     'dd_sum',
                'chunk_type':   Problem,
                'features':     {
                    'operator':     '+',
                    'operand1':     Variable('op1ndd'),
                    'operand2':     Variable('op2ndd')}},
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_dec_dig_in_ans',
                    'features':     {
                        'vproblem':     Variable('vproblem'),
                        'dd_from':      Variable('dd_sum'),
                        'dd_slot':      'answer'}})},
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {
                    'problem':      Variable('dd_sum')})}])

R_add_dd_skip = Rule( # If goal is add dec digs of ops to determine posn in ans, skip this step and use default posn
    name    = 'add_dd_skip',
    goal    = R_add_dd.goal,
    conds   = R_add_dd.conds,
    acts    = [{'action':       'pop_goal'},
               {'action':   'set_dec_dig_in_ans',
                'vproblem': Variable('vproblem')}])

## Fractions

# shared condition lists

fra_prob_conds = [ # general conditions for fraction problem
   {'target':   Variable('problem'),
    'pattern':  pat_prob},
   {'target':   Variable('operand1'),
    'pattern':  Number(features={
        'subtype':  'fraction',
        'num':      Variable('op1num'),
        'den':      Variable('op1den')})},
   {'target':   Variable('operand2'),
    'pattern':  Number(features={
        'subtype':  'fraction',
        'num':      Variable('op2num'),
        'den':      Variable('op2den')})}]
        
fra_unsolved_conds = fra_prob_conds + [ # conditions for unsolved fraction problem
   {'target':   Variable('answer'),
    'pattern':  'BLANK'}]

# shared action lists

def make_SAF_acts(probVarName): # set answer as fraction
    return([ 
       {'action':       'create_chunk',
        'variable':     'answer_in_progress',
        'chunk_type':   Number,
        'features':     {
            'subtype':      'fraction'}},
       {'action':       'set_feature',
        'chunk':        Variable(probVarName),
        'slot':         'answer',
        'value':        Variable('answer_in_progress')}])

def make_KDON_acts(probVarName):
   # do KDON (used in both KDON and CDON)
   return(make_SAF_acts(probVarName) + [
       # plan to simplify answer
       {'action':       'push_goal',
        'goal':         Goal('check_simplify', {
            'problem':      Variable(probVarName)})},       
       # plan to operate on numerators
       {'action':       'push_goal',
        'goal':         Goal('operate_nums', {
            'problem':      Variable(probVarName)})},
       # plan to pass through denominator
       {'action':       'push_goal',
        'goal':         Goal('pass_den', {
            'problem':      Variable(probVarName)})}])

def make_ONOD_acts(probVarName):
   # do ONOD (used in both ONOD and ICDM)
   return(make_SAF_acts(probVarName) + [
       # plan to simplify answer
       {'action':       'push_goal',
        'goal':         Goal('check_simplify', {
            'problem':      Variable(probVarName)})},       
       # plan to operate on numerators
       {'action':       'push_goal',
        'goal':         Goal('operate_nums', {
            'problem':      Variable(probVarName)})},
       # plan to operate on denominators
       {'action':       'push_goal',
        'goal':         Goal('operate_dens', {
            'problem':      Variable(probVarName)})}])

CNP_acts = [ # create copy of orig prob for purpose of modification
   # create new problem
   {'action':       'create_chunk',
    'variable':     'new_problem',
    'chunk_type':   Problem,
    'features':     {
        'operator':     Variable('operator'),
        'operand1':     Variable('operand1'),
        'operand2':     Variable('operand2')},
    'copy':         True},
   # push goal to set orig prob ans to new prob ans
   {'action':       'push_goal',
    'goal':         Goal('deferred_action', {
        'action':       'set_feature',
        'features':         {
            'chunk':        Variable('problem'),
            'slot':         'answer',
            'from':         Variable('new_problem'),
            'from_slot':    'answer'}})}]

# KDON: keep denominator, operate on numerators

R_KDON_AS = Rule( # KDON for addition and subtraction
    name    = 'KDON_AS',
    goal    = pat_solve_prob,
    conds   = fra_unsolved_conds + [
               ('==',
                {'comparand1':  Variable('op1den'),
                 'comparand2':  Variable('op2den')}),
               ('or',
                {'target':  Variable('operator'),
                 'pattern': '+'},
                {'target':  Variable('operator'),
                 'pattern': '-'})],
    acts    = make_KDON_acts('problem'))

R_KDON_OG = Rule( # KDON for any operation
    name    = 'KDON_OG',
    goal    = pat_solve_prob,
    conds   = R_KDON_AS.conds[:-1],
    acts    = R_KDON_AS.acts)

# CDON: convert denominator, operate on numerators

R_CDON_AS = Rule( # CDON for addition and subtraction
    name    = 'CDON_AS',
    goal    = pat_solve_prob,
    conds   = fra_unsolved_conds + [
               ('!=',
                {'comparand1':  Variable('op1den'),
                 'comparand2':  Variable('op2den')}),
               ('or',
                {'target':  Variable('operator'),
                 'pattern': '+'},
                {'target':  Variable('operator'),
                 'pattern': '-'})],
    acts    = CNP_acts + make_KDON_acts('new_problem') + [
               # plan to instantiate new prob by converting old prob to comm den
               {'action':       'push_goal',
                'goal':         Goal('convert_CD', {
                    'problem':      Variable('new_problem')})}])

R_CDON_OG = Rule( # CDON for any operation
    name    = 'CDON_OG',
    goal    = pat_solve_prob,
    conds   = R_CDON_AS.conds[:-1],
    acts    = R_CDON_AS.acts)

# ONOD: operate on numerators and denominators

R_ONOD_M = Rule( # ONOD for multiplication
    name    = 'ONOD_M',
    goal    = pat_solve_prob,
    conds   = fra_unsolved_conds + [
               {'target':  Variable('operator'),
                'pattern': '*'}],
    acts    = make_ONOD_acts('problem'))

R_ONOD_OG = Rule( # ONOD for any operation
    name    = 'ONOD_OG',
    goal    = pat_solve_prob,
    conds   = R_ONOD_M.conds[:-1],
    acts    = R_ONOD_M.acts)

# ICDM: invert and change division to multiplication

R_ICDM_D = Rule( # ICDM for division
    name    = 'ICDM_D',
    goal    = pat_solve_prob,
    conds   = fra_unsolved_conds + [
               {'target':  Variable('operator'),
                'pattern': ':'}],
    acts    = # create new problem
              CNP_acts + 
              # goals to solve new problem via ONOD
              make_ONOD_acts('new_problem') + [
              # goal to change division to multiplication (if applicable)
               {'action':       'push_goal',
                'goal':         Goal('div_to_mul', {
                    'problem':      Variable('new_problem')})},
              # goal to invert operand
              {'action':       'push_goal',
               'goal':         Goal('invert_operand', {
                    'problem':      Variable('new_problem')})}])

R_ICDM_OG = Rule( # ICDM for any operation
    name    = 'ICDM_OG',
    goal    = pat_solve_prob,
    conds   = R_ICDM_D.conds[:-1],
    acts    = R_ICDM_D.acts)

R_CROP_M = Rule( # Cross operate for multiplication
    name    = 'CROP_M',
    goal    = pat_solve_prob,
    conds   = R_ONOD_M.conds,
    acts    = # create new problem
              CNP_acts + 
              # goals to solve new problem via ONOD
              make_ONOD_acts('new_problem') + [
              # invert operand
               {'action':   'invert_operand',
                'problem':  Variable('new_problem')}])

# P2F: convert fraction problem with mixed and/or whole operands into "pure" fraction form

def make_OP2F_acts(probName, opName):
   new_goal_name = 'OP2F_goal_' + str(np.random.randint(1,10000))
   # create goals to convert operand into fraction
   return([
       # create conversion goal
       {'action':       'create_chunk',
        'variable':     new_goal_name,
        'chunk_type':   Goal,
        'features':     {
            'name':         'N2F',
            'number':       Variable(opName),
            'fraction':     None}},
       # push goal to set operand to conversion result
       {'action':       'push_goal',
        'goal':         Goal('deferred_action', {
            'action':       'set_feature',
            'features':     {
                'chunk':        Variable(probName),
                'slot':         opName,
                'from':         Variable(new_goal_name),
                'from_slot':    'fraction'}})},
       # push conversion goal
       {'action':       'push_goal',
        'goal':         Variable(new_goal_name)}])

R_P2F = Rule( # Convert mixed or whole operands to fractions
    name    = 'P2F',
    goal    = pat_solve_prob,
    conds   = [# problem is an unsolved problem with at least one mixed operand, or whole and a fraction operand
               {'target':       Variable('problem'),
                'pattern':      pat_prob},
               {'target':       Variable('answer'),
                'pattern':      'BLANK'},
               ('or',
                {'target':      Variable('operand1'),
                 'pattern':     Number(features={'subtype': 'mixed'})},
                {'target':      Variable('operand2'),
                 'pattern':     Number(features={'subtype': 'mixed'})},
                ('and',
                 {'target':     Variable('operand1'),
                  'pattern':    Number(features={'subtype': 'whole'})},
                 {'target':     Variable('operand2'),
                  'pattern':    Number(features={'subtype': 'fraction'})}),
                ('and',
                 {'target':     Variable('operand1'),
                  'pattern':    Number(features={'subtype': 'fraction'})},
                 {'target':     Variable('operand2'),
                  'pattern':    Number(features={'subtype': 'whole'})}))],
    acts    = CNP_acts + [
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', features={
                    'problem':      Variable('new_problem')})}] +
                make_OP2F_acts('new_problem', 'operand2') + 
                make_OP2F_acts('new_problem', 'operand1'))

# Execution rules
        
R_operate_nums = Rule( # Operate on numerators
    name    = 'operate_nums',
    goal    = Goal('operate_nums', {'problem': Variable('problem')}),
    conds   = fra_prob_conds,
    acts    = [{'action':   'pop_goal'}
              ] + make_SFA_acts(Variable('operator'), Variable('op1num'), Variable('op2num'), Variable('answer'), 'num'))

R_operate_dens = Rule( # Operate on denominators
    name    = 'operate_dens',
    goal    = Goal('operate_dens', {'problem': Variable('problem')}),
    conds   = fra_prob_conds,
    acts    = [{'action':   'pop_goal'}
              ] + make_SFA_acts(Variable('operator'), Variable('op1den'), Variable('op2den'), Variable('answer'), 'den'))

R_pass_den = Rule( # Pass denominator from operand into answer
    name    = 'pass_den',
    goal    = Goal('pass_den', {'problem': Variable('problem')}),
    conds   = fra_prob_conds,
    acts    = [{'action':   'pop_goal'},
               {'action':   'set_feature',
                'chunk':    Variable('answer'),
                'slot':     'den',
                'value':    Variable('op1den')}])

R_convert_CD = Rule( # Convert operands to common denominator
    name    = 'convert_CD',
    goal    = Goal('convert_CD', {'problem':  Variable('problem')}),
    conds   = fra_prob_conds,
    acts    = [{'action':   'pop_goal'}
              ] + make_SFA_acts('*', Variable('op2num'), Variable('op1den'), Variable('operand2'), 'num')
              + make_SFA_acts('*', Variable('op2den'), Variable('op1den'), Variable('operand2'), 'den')
              + make_SFA_acts('*', Variable('op1num'), Variable('op2den'), Variable('operand1'), 'num')
              + make_SFA_acts('*', Variable('op1den'), Variable('op2den'), Variable('operand1'), 'den'))

R_convert_CD_omit_nums = Rule( # Convert operands to common denominator but neglect to convert numerators
    name    = 'convert_CD_omit_nums',
    goal    = Goal('convert_CD', {'problem':  Variable('problem')}),
    conds   = fra_prob_conds,
    acts    = [{'action':   'pop_goal'}
              ] + make_SFA_acts('*', Variable('op2den'), Variable('op1den'), Variable('operand2'), 'den')
              + make_SFA_acts('*', Variable('op1den'), Variable('op2den'), Variable('operand1'), 'den'))

R_convert_CD_LCM = Rule( # Convert operands to common denominator by LCD procedure
    name    = 'convert_CD_LCM',
    goal    = Goal('convert_CD', {'problem':  Variable('problem')}),
    conds   = fra_prob_conds,
    acts    = # 1. Get LCM of op1den and op2den
              # 2. Convert operand1 in place to equivalent fraction with den equal to LCM
              # 3. Convert operand2 in place to equivalent fraction with den equal to LCM
              [{'action':   'pop_goal'},
               # Create and push goal to convert operand2 (3)
               {'action':       'create_chunk',
                'variable':     'convert_operand2_goal',
                'chunk_type':   Goal,
                'features':     {
                    'name':         'convert_fra_to_den',
                    'fraction':     Variable('operand2')}},
               {'action':        'push_goal',
                'goal':          Variable('convert_operand2_goal')},
               # Create and push goal to convert operand1 (2)
               {'action':       'create_chunk',
                'variable':     'convert_operand1_goal',
                'chunk_type':   Goal,
                'features':     {
                    'name':         'convert_fra_to_den',
                    'fraction':     Variable('operand1')}},
               {'action':        'push_goal',
                'goal':          Variable('convert_operand1_goal')},
               # Create and push goals to get LCM of op1den and op2den (1), then set 'denominator' slots in above goals to the result
               {'action':       'create_chunk',
                'variable':     'get_LCM_goal',
                'chunk_type':   Goal,
                'features':     {
                    'name':         'get_LCM',
                    'number1':      Variable('op1den'),
                    'number2':      Variable('op2den')}},
               {'action':        'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('convert_operand2_goal'),
                        'slot':         'denominator',
                        'from':         Variable('get_LCM_goal'),
                        'from_slot':    'LCM'}})},
               {'action':        'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('convert_operand1_goal'),
                        'slot':         'denominator',
                        'from':         Variable('get_LCM_goal'),
                        'from_slot':    'LCM'}})},
               {'action':        'push_goal',
                'goal':          Variable('get_LCM_goal')}])

R_convert_fra_to_den = Rule( # Convert fraction (in place) to equivalent fraction with given denominator
    name    = 'convert_fra_to_den',
    goal    = Goal('convert_fra_to_den',  features={
                    'fraction':     Variable('fraction'),
                    'denominator':  Variable('targ_den')}),
    conds   = [{'target':       Variable('fraction'),
                'pattern':      Number(features={
                    'subtype':      'fraction',
                    'num':          Variable('src_num'),
                    'den':          Variable('src_den')})}],
    acts    = # 1. Calculate quotient of targ_den / src_den
              # 2. Calculate product of src_num * quotient
              # 3. Set fraction num to product
              # 4. Set fraction den to targ_den
              [{'action':       'pop_goal'},
               # Create problem representing quotient
               {'action':       'create_chunk',
                'variable':     'quotient',
                'chunk_type':   Problem,
                'features':     {
                    'operator':     ':',
                    'operand1':     Variable('targ_den'),
                    'operand2':     Variable('src_den')}},
               # Create problem representing product
               {'action':       'create_chunk',
                'variable':     'product',
                'chunk_type':   Problem,
                'features':     {
                    'operator':     '*',
                    'operand1':     Variable('src_num')}},
               # Create goal to set fraction den to targ_den (4)
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('fraction'),
                        'slot':         'den',
                        'value':        Variable('targ_den')}})},
               # Create goal to set fraction num to answer in product (3)
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('fraction'),
                        'slot':         'num',
                        'from':         Variable('product'),
                        'from_slot':    'answer'}})},
               # Create goal to solve product (2)
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {'problem': Variable('product')})},
               # Create goals to solve quotient and set product operand2 to its answer (1)
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('product'),
                        'slot':         'operand2',
                        'from':         Variable('quotient'),
                        'from_slot':    'answer'}})},
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {'problem': Variable('quotient')})}])

R_get_LCM = Rule( # Get LCM and store it in 'LCM' slot of current goal
    name    = 'get_LCM',
    goal    = Goal('get_LCM', features={
                    'number1':      Variable('number1'),
                    'number2':      Variable('number2')}),
    conds   = [],
    acts    = [{'action':   'pop_goal'},
               # use primitive action to determine LCM (in future need realistic LCM procedure)
               {'action':   'get_LCM',
                'number1':  Variable('number1'),
                'number2':  Variable('number2'),
                'chunk':    Variable('curr_goal'),
                'slot':     'LCM'}])

R_invert_op2 = Rule( # Invert second operand
    name    = 'invert_op2',
    goal    = Goal('invert_operand', {'problem':  Variable('problem')}),
    conds   = [],
    acts    = [{'action':   'pop_goal'},
               {'action':   'invert_operand',
                'problem':  Variable('problem'),
                'operand':  'operand2'}])

R_invert_rand = Rule( # Invert random operand
    name    = 'invert_rand',
    goal    = Goal('invert_operand', {'problem':  Variable('problem')}),
    conds   = [],
    acts    = [{'action':   'pop_goal'},
               {'action':   'invert_operand',
                'problem':  Variable('problem')}])

R_invert_fail = Rule( # Fail to invert operand
    name    = 'invert_fail',
    goal    = Goal('invert_operand', {'problem':  Variable('problem')}),
    conds   = [],
    acts    = R_invert_op2.acts[:-1])

R_div_to_mul = Rule( # Change division to multiplication
    name    = 'div_to_mul',
    goal    = Goal('div_to_mul', {'problem': Variable('problem')}),
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('operator'),
                'pattern':  ':'}],
    acts    = [{'action':   'pop_goal'},
               {'action':   'set_feature',
                'chunk':    Variable('problem'),
                'slot':     'operator',
                'value':    '*'}])

R_div_to_mul_denied = Rule( # Drop div_to_mul goal if inapplicable
    # It's kludgy to have a special-purpose "denied" rule for such a specific goal
    # Probably more realistic to have general "replace x with y" goal
    # with a pair of rules that do it if x is present and drop it if not
    name    = 'div_to_mul_denied',
    goal    = Goal('div_to_mul', {'problem': Variable('problem')}),
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               ('not',
                {'target':  Variable('operator'),
                 'pattern': ':'})],
    acts    = [{'action':   'pop_goal'}])

R_N2F_fra = Rule( # Convert fraction to fraction
    name    = 'N2F_fra',
    goal    = Goal('N2F', features={'number': Variable('number'), 'fraction': 'BLANK'}),
    conds   = [{'target':       Variable('number'),
                'pattern':      pat_fraction}],
    acts    = [{'action':       'pop_goal'},
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'fraction',
                'value':        Variable('number')}])
            
R_N2F_wn = Rule( # Convert whole number to fraction
    name    = 'N2F_wn',
    goal    = Goal('N2F', features={'number': Variable('number'), 'fraction': 'BLANK'}),
    conds   = [{'target':       Variable('number'),
                'pattern':      pat_whole}],
    acts    = [{'action':       'pop_goal'},
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'fraction',
                'value':        Number(features={
                    'subtype':      'fraction', 
                    'num':          Variable('number'), 
                    'den':          Number(1)})}])
 
R_N2F_mix = Rule( # Convert mixed number to fraction
    name    = 'N2F_mix',
    goal    = Goal('N2F', features={'number': Variable('number'), 'fraction': 'BLANK'}),
    conds   = [{'target':       Variable('number'),
                'pattern':      Number(features={
                    'subtype':      'mixed',
                    'whole':        Variable('whole_part'),
                    'fraction':     Variable('fraction_part')})},
                {'target':      Variable('fraction_part'),
                 'pattern':     Number(features={
                    'subtype':      'fraction',
                    'num':          Variable('numerator'),
                    'den':          Variable('denominator')})}],
    acts    = [{'action':       'pop_goal'},
               # create fraction to store result and set its denom to fraction part denom
               {'action':       'create_chunk',
                'variable':     'conversion_result',
                'chunk_type':   Number,
                'features':     {'subtype': 'fraction'}},
               {'action':       'set_feature',
                'chunk':        Variable('curr_goal'),
                'slot':         'fraction',
                'value':        Variable('conversion_result')},
               {'action':       'set_feature',
                'chunk':        Variable('conversion_result'),
                'slot':         'den',
                'value':        Variable('denominator')},
               # create problem representing product of whole part and fraction part denominator
               {'action':       'create_chunk',
                'variable':     'product',
                'chunk_type':   Problem,
                'features':     {
                    'operator':     '*',
                    'operand1':     Variable('whole_part'),
                    'operand2':     Variable('denominator')}},
               # create problem representing sum of above product and fraction part numerator
               {'action':       'create_chunk',
                'variable':     'sum',
                'chunk_type':   Problem,
                'features':     {
                    'operator':     '+',
                    'operand2':     Variable('numerator')}},
               # create goal to set result numerator equal to above sum
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('conversion_result'),
                        'slot':         'num',
                        'from':         Variable('sum'),
                        'from_slot':    'answer'}})},
               # create goal to calculate above sum
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {'problem': Variable('sum')})},
               # create goal to set operand1 of above sum to above product
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('sum'),
                        'slot':         'operand1',
                        'from':         Variable('product'),
                        'from_slot':    'answer'}})},
               # create goal to calculate above product
               {'action':       'push_goal',
                'goal':         Goal('solve_problem', {'problem': Variable('product')})}])

R_check_simplify = Rule( # Check whether answer to fraction problem must be simplified
    name    = 'check_simplify',
    goal    = Goal('check_simplify', {'problem': Variable('problem')}),
    conds   = [{'target':    Variable('problem'),
                'pattern':   pat_prob},
               {'target':    Variable('answer'),
                'pattern':   Number(features={
                    'subtype':  'fraction',
                    'num':      Variable('ans_num'),
                    'den':      Variable('ans_den')})},
               ('or',
                {'target':   Variable('ans_num'),
                 'pattern':  Number(features={'subtype': 'whole'})},
                {'target':   Variable('ans_num'),
                 'pattern':  Number(features={'subtype': 'negint'})}),
               ('or',
                {'target':   Variable('ans_den'),
                 'pattern':  Number(features={'subtype': 'whole'})},
                {'target':   Variable('ans_den'),
                 'pattern':  Number(features={'subtype': 'negint'})})],
    acts    = [{'action':   'pop_goal'},
               # Create goals to get GCD of ans_num and ans_den, then do simplification using the GCD
               {'action':       'create_chunk',
                'variable':     'get_GCD_goal',
                'chunk_type':   Goal,
                'features':     {
                    'name':         'get_GCD',
                    'number1':      Variable('ans_num'),
                    'number2':      Variable('ans_den')}},
               {'action':       'create_chunk',
                'variable':     'simplify_fraction_goal',
                'chunk_type':   Goal,
                'features':     {
                    'name':         'simplify_fraction',
                    'fraction':     Variable('answer')}},
               # Push do_simplify goal
               {'action':        'push_goal',
                'goal':          Variable('simplify_fraction_goal')},
               # Push goal to set GCD in do_simplify goal to GCD in get_GCD goal
               {'action':       'push_goal',
                'goal':         Goal('deferred_action', {
                    'action':       'set_feature',
                    'features': {
                        'chunk':        Variable('simplify_fraction_goal'),
                        'slot':         'GCD',
                        'from':         Variable('get_GCD_goal'),
                        'from_slot':    'GCD'}})},
               # Push goal to get GCD
               {'action':       'push_goal',
                'goal':         Variable('get_GCD_goal')}])

R_skip_simplify = Rule( # Skip simplifying
    name    = 'skip_simplify',
    goal    = R_check_simplify.goal,
    conds   = [],
    acts    = [{'action':   'pop_goal'}])

R_get_GCD = Rule( # Get GCD and store it in 'GCD' slot of current goal
    name    = 'get_GCD',
    goal    = Goal('get_GCD', features={
                    'number1':      Variable('number1'),
                    'number2':      Variable('number2')}),
    conds   = [],
    acts    = [{'action':   'pop_goal'},
               # use primitive action to determine GCD (in future need realistic GCD procedure)
               {'action':   'get_GCD',
                'number1':  Variable('number1'),
                'number2':  Variable('number2'),
                'chunk':    Variable('curr_goal'),
                'slot':     'GCD'}])

R_simplify_fraction = Rule( # If GCD>1, convert fraction (in place) to equivalent fraction in lowest terms
    name    = 'simplify_fraction',
    goal    = Goal('simplify_fraction', features={
                    'fraction': Variable('fraction'),
                    'GCD':      Variable('GCD')}),
    conds   = [('<',
                {'comparand1':  Number(1),
                 'comparand2':  Variable('GCD')}),
               {'target':   Variable('fraction'),
                'pattern':  Number(features={
                    'subtype':  'fraction',
                    'num':      Variable('num'),
                    'den':      Variable('den')})}],
    acts    = [{'action':   'pop_goal'}] +
              make_SFA_acts(':', Variable('den'), Variable('GCD'), Variable('fraction'), 'den') +
              make_SFA_acts(':', Variable('num'), Variable('GCD'), Variable('fraction'), 'num'))
              
R_cannot_simplify = Rule( # If GCD is 1 or 0, remove simplify fraction goal
    name    = 'cannot_simplify',
    goal    = Goal('simplify_fraction', features={
                    'fraction': Variable('fraction'),
                    'GCD':      Variable('GCD')}),
    conds   = [('or',
                ('==',
                 {'comparand1': Variable('GCD'),
                  'comparand2': Number(1)}),
                ('==',
                 {'comparand1': Variable('GCD'),
                  'comparand2': Number(0)}))],
    acts    = [{'action':   'pop_goal'}])

### Models

all_rules   = [

    ## base rules
    R_deferred_action,
    R_finish_problem,
    R_op1_none_to_zero,
    R_op2_none_to_zero,
    
    ## arithmetic principles
    R_larger_first_add_mul,
    R_add_to_zero,
    R_sub_zero_from,
    R_mul_by_zero, 
    R_mul_by_one,

    ## addition and multiplication of small whole numbers
    R_retrieve,     # fact retrieval
    R_acc_once,     # accumulator algorithm
    R_acc_skip,
    R_acc_extra,
    R_acc_end,
    R_add_count,    # addition using accumulator
    R_acc_count,
    R_mul_rep_add,  # multiplication using accumulator
    R_acc_add,
    # R_add_fact,
    # R_mul_fact,

    ## subtraction and division of small whole numbers
    R_sub_fact,
    R_sub_LbS,          # error
    R_div_calculator,
    R_div_LbS,          # error
    R_div_LbS_drop_rem, # error
    R_div_drop_rem,     # error

    ## multidigit arithmetic (except division)
    # list aggregation
    R_LA_start, 
    R_LA_two, 
    R_LA_set, 
    R_LA_calc, 
    R_LA_skip, 
    R_LA_zero, 
    R_LA_end,
    # change horizontal to vertical format and choose a vertical algorithm
    R_H2V_WN, 
    R_align_right, 
    R_VA_start,
    # valg for add/sub
    R_choose_VAS_AS,
    R_VA_next_calc, 
    R_VA_lone_carry, 
    R_VS_next_calc, 
    R_VS_borrow_start, 
    R_VS_borrow_to, 
    R_VS_borrow_from_nonzero, 
    R_VS_borrow_from_zero,
    R_VS_next_calc_swap,    # error
    R_VS_borrow_from_fail,  # error
    R_VAS_finish, 
    R_VAS_end_calc, 
    R_VAS_do_carry, 
    R_VAS_add_carry,
    R_VAS_shift_attn,
    # valg for mul
    R_choose_VM_M,
    R_VM_next_calc, 
    R_VM_lone_carry, 
    R_VM_finish_sd,
    R_VM_end_calc, 
    R_VM_do_carry, 
    R_VM_add_carry,
    R_VM_shift1,
    R_VM_shift2, 
    R_VM_new_row, 
    R_VM_add_parts,
    R_VM_shift2_no_zeros,   # error

    ## decimal arithmetic (except division)
    # strategy rules
    R_ADBD_AS,
    R_ADBD_OG,              # error
    R_ARAD_M,
    R_ARAD_OG,              # error
    # execution rules
    R_align_dec_no_zeros,
    R_align_dec_app_zeros,
    R_bring_dec,
    R_add_dd,
    R_add_dd_skip,          # error

    ## fraction arithmetic
    # converting whole/mixed numbers to fractions
    R_P2F,
    R_N2F_fra, 
    R_N2F_wn, 
    R_N2F_mix,
    # strategy rules
    R_KDON_AS, 
    R_CDON_AS, 
    R_ONOD_M, 
    R_ICDM_D,
    # execution rules
    R_operate_nums, 
    R_operate_dens, 
    R_pass_den, 
    R_convert_CD,
    R_convert_CD_LCM,
    R_get_LCM,
    R_convert_fra_to_den,
    R_invert_op2, 
    R_div_to_mul, 
    # overgeneralized strategies
    R_KDON_OG, 
    R_CDON_OG, 
    R_ONOD_OG, 
    R_ICDM_OG, 
    R_CROP_M,
    # execution errors
    R_convert_CD_omit_nums, 
    R_invert_rand, 
    R_invert_fail, 
    R_div_to_mul_denied,
    # simplifying answer
    R_check_simplify,
    R_skip_simplify,
    R_get_GCD,
    R_simplify_fraction,
    R_cannot_simplify
    ]

# Create rules for rational arithmetic without single digit arithmetic errors

SD_add_mul_rules_realistic = [
    R_retrieve,
    R_add_count,
    R_mul_rep_add ]
    
SD_add_mul_rules_perfect = [
    R_add_fact,
    R_mul_fact ]
    
RA_rules = SD_add_mul_rules_perfect + [
    rule for rule in all_rules
    if rule.name not in [r.name for r in SD_add_mul_rules_realistic]]

# # Example 1
# M = UMA(rules=all_rules)
# M.run("4+2", verbose=True, learn=True)

# # Example 2
# M = UMA(rules=all_rules)
# teachRetrieval(M)
# M.run("4*2", verbose=True, learn=True)

# # Example 3
# M = UMA(rules=RA_rules)
# M.run("3/5+1/4", verbose=True, learn=True)

# # Example 4
# M = UMA(rules=RA_rules)
# M.run("2.4*1.2", verbose=True, learn=True)
