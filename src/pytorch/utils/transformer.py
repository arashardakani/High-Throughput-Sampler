import csv
from pysat.formula import CNF
from itertools import product
import sympy
from sympy import Or, And, Not, Symbol
from sympy.logic.boolalg import Xor, Xnor
from sympy import symbols, Symbol
from sympy.logic import simplify_logic
import re
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import time

def is_conjunction_of_literals(expr):
    # Base case: If it's a single literal (either a variable or its negation)
    if isinstance(expr, Symbol) or (expr.func == Not and isinstance(expr.args[0], Symbol)):
        return True
    
    # Check if the expression is an AND operation
    if expr.func == And:
        # Recursively check if all arguments are either literals or AND of literals
        return all(is_conjunction_of_literals(arg) for arg in expr.args)
    
    return False

def is_disjunction_of_literals(expr):
    # Base case: If it's a single literal (either a variable or its negation)
    if isinstance(expr, Symbol) or (expr.func == Not and isinstance(expr.args[0], Symbol)):
        return True
    
    # Check if the expression is an AND operation
    if expr.func == Or:
        # Recursively check if all arguments are either literals or AND of literals
        return all(is_disjunction_of_literals(arg) for arg in expr.args)
    
    return False

def extract_symbols_with_negation(expr):
    symbols = []
    
    # If the expression is a single symbol, return it
    if isinstance(expr, Symbol):
        return [expr]
    
    # If the expression is a negation (~ or Not), append the negated symbol
    if expr.func == Not and isinstance(expr.args[0], Symbol):
        return [~expr.args[0]]
    
    # If the expression is a compound expression (e.g., And, Or), recursively process its arguments
    for arg in expr.args:
        symbols.extend(extract_symbols_with_negation(arg))
    
    return symbols

def simplify_clauses(clauses, output_var, hardcoded_inputs):
    
    hardcoded_inputs_abs = [abs(in_var) for in_var in hardcoded_inputs]
    # Create symbols for variables
    variables = set()
    for clause in clauses:
        for var in clause:
            if abs(var) != abs(output_var):
                variables.add(abs(var))

    symbols_dict = {var: symbols(f'x{var}') for var in variables}
    output_symbol = symbols(f'x{abs(output_var)}')
    # Convert clauses to boolean expressions
    boolean_clauses = []
    for clause in clauses:
        if output_var not in clause:
            terms = []
            cond = {}
            for var in clause:
                
                if abs(var) == abs(output_var):
                    continue  # Skip output variable
                term = symbols_dict[abs(var)]
                if var < 0:
                    term = Not(term)
                if abs(var) in hardcoded_inputs_abs:
                    if abs(var) in hardcoded_inputs:
                        cond[symbols(f'x{abs(var)}')] = True
                    elif -abs(var) in hardcoded_inputs:
                        cond[symbols(f'x{abs(var)}')] = False
                terms.append(term)
            if cond:
                boolean_clauses.append(Or(*terms).subs(cond))
            else:
                
                boolean_clauses.append(Or(*terms))
    simplified_expression = simplify_logic(And(*boolean_clauses)) 
    return simplified_expression 



def replace_boolean_operators(expression):
    if '^' in expression:
        not_gate = False
        if '~' in expression:
            not_gate = True
            expression = re.sub(r'\~', '', expression)
            expression = re.sub(r'\(', '', expression)
            expression = re.sub(r'\)', '', expression)
        expression = re.sub(r'\^', ',', expression)
        terms = [term.strip() for term in expression.split(',')]
        expression = 'XOR(' + ', '.join(terms) + ')'
        if not_gate:
            expression = 'NOT(' + expression + ')'
    # Base case: if the expression does not contain any parentheses
    elif '(' not in expression:
        if '&' in expression:
            expression = re.sub(r'&', ',', expression)
            terms = [term.strip() for term in expression.split(',')]
            converted_terms = [re.sub(r'~(x\d+)', r'NOT(\1)', term) for term in terms]
            expression = 'AND(' + ', '.join(converted_terms) + ')'
        elif '|' in expression:
            expression = re.sub(r'\|', ',', expression)
            terms = [term.strip() for term in expression.split(',')]
            converted_terms = [re.sub(r'~(x\d+)', r'NOT(\1)', term) for term in terms]
            expression = 'OR(' + ', '.join(converted_terms) + ')'
        else:
            expression = re.sub(r'~(x\d+)', r'NOT(\1)', expression) 
    
    # Recursive case: process the most nested expressions first
    elif '(' in expression:
        
        if (') |' in expression) or ('| (' in expression):
            
            expression = re.sub(r'&', ',', expression)
            expression = re.sub(r'\|', ',', expression)
            expression = re.sub(r'\(', ' AND(', expression)
            expression = re.sub(r'~(x\d+)', r'NOT(\1)', expression)
            expression = 'OR(' + expression + ')'
        elif (') &' in expression) or ('& (' in expression):
            
            expression = re.sub(r'&', ',', expression)
            expression = re.sub(r'\|', ',', expression)
            expression = re.sub(r'\(', 'OR(', expression)
            expression = re.sub(r'~(x\d+)', r'NOT(\1)', expression)
            expression = 'AND(' + expression + ')'

    return expression


def transformer(cnf):
    clause_groups = {}
    clause_gate = {}
    clause_output = {}

    primary_inputs = []
    primary_outputs = []
    intermadiate_vars = []
    hardcoded_inputs = []


    vars = []
    End = False 
    cnt = 0
    idx = 0
    no_groups = 0
    single_vars = []
    while not End:
        
        offset = 0

        clause_groups[no_groups] = []
        clause_gate[no_groups] = []
        clause_output[no_groups] = []
        group_clauses = []
        valid_candidates = []
        gate_solution = False

        current_clause = cnf.clauses[idx]
        group_vars = [abs(var) for var in cnf.clauses[idx]]
        hardcoded_inputs_abs = [abs(in_var) for in_var in hardcoded_inputs]

        if (idx < len(cnf.clauses)):
            sub_groups = True  
        else:
            sub_groups = False
            End = True
        while(sub_groups):
            

            current_clause = cnf.clauses[idx + offset]
            group_clauses.append(current_clause)
            current_vars = [abs(var) for var in current_clause]
            
            # print(current_vars, group_vars, current_clause)
            # print('hardcoded_inputs',hardcoded_inputs)

            group_vars = list(set(group_vars + current_vars))
            for var in group_vars:
                if var not in hardcoded_inputs_abs + primary_inputs + intermadiate_vars or ((len(group_clauses) == 1) and (len(group_vars) == 1)):
                    expression = simplify_clauses(group_clauses, var, [])
                    complement_of_expression = simplify_clauses(group_clauses, -var, [])
                    isinstance_expression = False if (expression == False) or (expression == True) else True
                    isinstance_complement_of_expression = False if (complement_of_expression == False) or (complement_of_expression == True) else True
                    flag = False if ((not isinstance_expression and isinstance_complement_of_expression) or \
                            (isinstance_expression and not isinstance_complement_of_expression)) \
                            else sympy.logic.boolalg.Boolean.equals(expression, Not(complement_of_expression))

                    if flag == True:
                        if var not in primary_inputs + intermadiate_vars or not isinstance_expression:
                            sub_groups = False
                            gate_solution = True
                            valid_candidates.append(var)
                    elif (expression == False) or (complement_of_expression == False):
                        sub_groups = False
                        valid_candidates.append(var)
                        break
            
            if not sub_groups:
                if gate_solution:
                    idx = idx + offset + 1
                    if len(group_vars) == 1:
                        clause_output[no_groups] = group_clauses[0][0]
                        clause_gate[no_groups] = simplify_clauses(group_clauses, group_clauses[0][0], [])
                        clause_groups[no_groups] = group_clauses
                        hardcoded_inputs.append(group_clauses[0][0])
                        hardcoded_inputs_abs.append(abs(group_clauses[0][0]))
                        for var in group_vars:
                            if var not in primary_inputs and var not in intermadiate_vars:
                                primary_inputs.append(var)
                    else:
                        variables = sorted(valid_candidates)
                        output_var = variables[-1]
                        input_var = [var for var in group_vars if var != output_var]
                        clause_output[no_groups] = output_var
                        clause_gate[no_groups] = simplify_clauses(group_clauses, output_var, hardcoded_inputs)  #[]
                        clause_groups[no_groups] = group_clauses
                        if clause_gate[no_groups] == True:
                                hardcoded_inputs.append(output_var)
                                hardcoded_inputs_abs.append(output_var)
                                clause_output[no_groups] = output_var
                        elif clause_gate[no_groups] == False:
                                clause_gate[no_groups] = True
                                clause_output[no_groups] = -output_var
                                hardcoded_inputs.append(-output_var)
                                hardcoded_inputs_abs.append(output_var)
                        else:
                            intermadiate_vars.append(output_var)

                        for var in input_var:
                            if var not in primary_inputs and var not in intermadiate_vars:
                                primary_inputs.append(var)
                    no_groups += 1
                else:
                    if simplify_clauses(group_clauses, valid_candidates[0], hardcoded_inputs) == False:
                        hardcoded_inputs.append(-valid_candidates[0])
                        hardcoded_inputs_abs.append(valid_candidates[0])
                        clause_output[no_groups] = -valid_candidates[0]
                        clause_gate[no_groups] = True
                        clause_groups[no_groups] = group_clauses
                    else:
                        hardcoded_inputs.append(valid_candidates[0])
                        hardcoded_inputs_abs.append(valid_candidates[0])
                        clause_output[no_groups] = valid_candidates[0]
                        clause_gate[no_groups] = True
                        clause_groups[no_groups] = group_clauses
                    no_groups += 1

                    valid_candidates = []
                    sol_flag = True
                    while sol_flag:
                        sol_flag = False
                        for var in group_vars:
                            if var not in hardcoded_inputs_abs:
                                expression = simplify_clauses(group_clauses, var, hardcoded_inputs)
                                complement_of_expression = simplify_clauses(group_clauses, -var, hardcoded_inputs)
                                if (expression == False):
                                    hardcoded_inputs.append(-var)
                                    hardcoded_inputs_abs.append(var)
                                    clause_output[no_groups] = -var
                                    clause_gate[no_groups] = True
                                    clause_groups[no_groups] = group_clauses
                                    sol_flag = True
                                    no_groups += 1
                                elif (complement_of_expression == False):
                                    hardcoded_inputs.append(var)
                                    hardcoded_inputs_abs.append(var)
                                    clause_output[no_groups] = var
                                    clause_gate[no_groups] = True
                                    clause_groups[no_groups] = group_clauses
                                    sol_flag = True
                                    no_groups += 1
                    if all(abs(var) in hardcoded_inputs_abs for var in group_vars):
                        idx = idx + offset + 1
                
            else:
                if all(abs(var) not in hardcoded_inputs_abs for var in group_vars):
                    if all(abs(var) not in group_vars for var in cnf.clauses[idx + offset + 1]): 
                        sub_groups = False
                        idx = idx + offset + 1

                        clause_gate[no_groups] = simplify_clauses(group_clauses, 0, hardcoded_inputs)
                        clause_groups[no_groups] = group_clauses
                        input_var = sorted(group_vars)
                        
                        clause_output[no_groups] = 0
                        primary_inputs.extend([var for var in input_var if var not in intermadiate_vars and var not in primary_inputs])
                        no_groups += 1
                elif all(abs(var) in hardcoded_inputs_abs for var in group_vars):
                    sub_groups = False
                    idx = idx + offset + 1
                elif (idx + offset + 1) == len(cnf.clauses):
                    sub_groups = False
                    idx = idx + offset + 1
                    sol_flag = True
                    
                    while sol_flag:
                        sol_flag = False
                        for var in group_vars:
                            if var not in hardcoded_inputs_abs:
                                expression = simplify_clauses(group_clauses, var, hardcoded_inputs)
                                complement_of_expression = simplify_clauses(group_clauses, -var, hardcoded_inputs)
                                if (expression == False):
                                    hardcoded_inputs.append(-var)
                                    hardcoded_inputs_abs.append(var)
                                    clause_output[no_groups] = -var
                                    clause_gate[no_groups] = True
                                    clause_groups[no_groups] = group_clauses
                                    sol_flag = True
                                    no_groups += 1
                                elif (complement_of_expression == False):
                                    hardcoded_inputs.append(var)
                                    hardcoded_inputs_abs.append(var)
                                    clause_output[no_groups] = var
                                    clause_gate[no_groups] = True
                                    clause_groups[no_groups] = group_clauses
                                    sol_flag = True
                                    no_groups += 1
                    if any(abs(var) not in hardcoded_inputs_abs for var in group_vars):
                        clause_gate[no_groups] = simplify_clauses(group_clauses, 0, hardcoded_inputs)
                        clause_groups[no_groups] = group_clauses
                        input_var = sorted(group_vars)
                        
                        clause_output[no_groups] = 0
                        primary_inputs.extend([var for var in input_var if var not in intermadiate_vars and var not in primary_inputs])
                        no_groups += 1

                elif all(abs(var) not in group_vars for var in cnf.clauses[idx + offset + 1]):
                    sub_groups = False
                    idx = idx + offset + 1

                    sol_flag = True
                    
                    while sol_flag:
                        sol_flag = False
                        for var in group_vars:
                            if var not in hardcoded_inputs_abs:
                                expression = simplify_clauses(group_clauses, var, hardcoded_inputs)
                                complement_of_expression = simplify_clauses(group_clauses, -var, hardcoded_inputs)
                                if (expression == False):
                                    hardcoded_inputs.append(-var)
                                    hardcoded_inputs_abs.append(var)
                                    clause_output[no_groups] = -var
                                    clause_gate[no_groups] = True
                                    clause_groups[no_groups] = group_clauses
                                    sol_flag = True
                                    no_groups += 1
                                elif (complement_of_expression == False):
                                    hardcoded_inputs.append(var)
                                    hardcoded_inputs_abs.append(var)
                                    clause_output[no_groups] = var
                                    clause_gate[no_groups] = True
                                    clause_groups[no_groups] = group_clauses
                                    sol_flag = True
                                    no_groups += 1
                    if any(abs(var) not in hardcoded_inputs_abs for var in group_vars):
                        clause_gate[no_groups] = simplify_clauses(group_clauses, 0, hardcoded_inputs)
                        clause_groups[no_groups] = group_clauses
                        input_var = sorted(group_vars)
                        
                        clause_output[no_groups] = 0
                        primary_inputs.extend([var for var in input_var if var not in intermadiate_vars and var not in primary_inputs])
                        no_groups += 1


            offset += 1
            if idx >= len(cnf.clauses):
                End = True

    exist = True
    hardcoded_inputs_abs = [abs(x) for x in hardcoded_inputs]
    while exist:
        exist = False
        for i in range(len(clause_output)):
            vars = []
            if clause_gate[i] and not (clause_gate[i] == True or  clause_gate[i] == False):
                for clause in clause_groups[i]:
                    for var in clause:
                        if abs(var) not in vars:
                            vars.append(abs(var))

                if any(abs(var) in hardcoded_inputs_abs for var in vars):
                    clause_gate[i] = simplify_clauses(clause_groups[i], clause_output[i] if clause_output[i] else 0, hardcoded_inputs)
                    # print(clause_gate[i])
                    if clause_gate[i] == True and not (clause_output[i] == 0):
                        hardcoded_inputs.append(clause_output[i])
                        hardcoded_inputs_abs.append(abs(clause_output[i]))
                        exist = True
                    elif clause_gate[i] == False and not (clause_output[i] == 0):
                        hardcoded_inputs.append(-clause_output[i])
                        hardcoded_inputs_abs.append(abs(clause_output[i]))
                        exist = True
                symbols_list = extract_symbols_with_negation(clause_gate[i])
                if len(symbols_list) > 4:
                    if is_conjunction_of_literals(clause_gate[i]) and (clause_output[i] in hardcoded_inputs):
                        
                        clause_gate[i] = []
                        
                        for var in vars:
                            if symbols(f'x{abs(var)}') in symbols_list:
                                hardcoded_inputs.append(var)
                                hardcoded_inputs_abs.append(var)
                            elif ~symbols(f'x{abs(var)}') in symbols_list:
                                hardcoded_inputs.append(-var)
                                hardcoded_inputs_abs.append(var)
                        clause_groups[i] = []
                        clause_output[i] = []
                        exist = True
                    elif is_disjunction_of_literals(clause_gate[i]) and (-clause_output[i] in hardcoded_inputs):
                        clause_gate[i] = []
                        for var in vars:
                            if symbols(f'x{abs(var)}') in symbols_list:
                                hardcoded_inputs.append(-var)
                                hardcoded_inputs_abs.append(var)
                            elif ~symbols(f'x{abs(var)}') in symbols_list:
                                hardcoded_inputs.append(var)
                                hardcoded_inputs_abs.append(var)
                        clause_groups[i] = []
                        clause_output[i] = []
                        exist = True

    clause_gate_list = list(clause_gate.values())
    for index, element in enumerate(clause_gate_list):
        if element:
            if not (element == True or element == False):
                if element in clause_gate_list[:index]:
                    idx = clause_gate_list[:index].index(element)
                    clause_gate[index] = symbols(f'x{abs(clause_output[idx])}')

    output_list = [] 
    for i in range(len(clause_groups)):
        if clause_gate[i]:
            if not (clause_gate[i] == True or clause_gate[i] == False):
                output_list.append(abs(clause_output[i]))
    
    hardcoded_inputs = list(set(hardcoded_inputs))
    hardcoded_inputs_list = set([abs(x) for x in hardcoded_inputs])
    uncommon_elements = list((hardcoded_inputs_list - set(output_list)))  #list((hardcoded_inputs_list - set(output_list)) | (set(output_list) - hardcoded_inputs_list)) 
    inputs = [str(symbols(f'x{abs(term)}')) for term in primary_inputs] + [str(symbols(f'x{abs(term_)}')) for term_ in uncommon_elements]
    inputs = list(set(inputs))

    outputs = [str(symbols(f'x{abs(term)}')) if term > 0 else str(symbols(f'NOT(x{abs(term)})')) for term in hardcoded_inputs]
    variables = [str(symbols(f'x{abs(term)}')) for term in set([abs(term) for term in sum(cnf.clauses, [])])]


    return inputs, outputs, variables, clause_gate, clause_output


def generate_pytorch_model(inputs, outputs, variables, clause_gate, clause_output):
    class_definition = f"class DUT(nn.Module):\n" \
                        f"    def __init__(self, batch_size, device):\n" \
                        f"        super().__init__()\n"
    class_definition += f"        self.batch_size = batch_size\n"
    class_definition += f"        self.device = device\n"
    class_definition += "\n"
    class_definition += f"    def forward(self, inputs):\n"
    for i, term in enumerate(inputs, start=0):
        class_definition += f"        {term} = inputs[{i}]\n"

    gate_constraint = []
    for i in range(len(clause_output)):
        if (clause_output[i] == 0) and not (clause_gate[i] == True or clause_gate[i] == False):
            gate_constraint.append(replace_boolean_operators(str(clause_gate[i])))
        elif clause_gate[i] and not (clause_gate[i] == True or  clause_gate[i] == False):
            num_vars = list(clause_gate[i].free_symbols)
            if len(num_vars) == 2:
                if sympy.logic.boolalg.Boolean.equals(Xor(num_vars[0], num_vars[1]), clause_gate[i]):
                    class_definition += f"        {str(symbols(f'x{abs(clause_output[i])}')) + ' = ' + replace_boolean_operators(str(Xor(num_vars[0], num_vars[1])))}\n"
                elif sympy.logic.boolalg.Boolean.equals(Xnor(num_vars[0], num_vars[1]), clause_gate[i]):
                    class_definition += f"        {str(symbols(f'x{abs(clause_output[i])}')) + ' = ' + replace_boolean_operators(str(Xnor(num_vars[0], num_vars[1])))}\n"
                else:
                    class_definition += f"        {str(symbols(f'x{abs(clause_output[i])}')) + ' = ' + replace_boolean_operators(str(clause_gate[i]))}\n"
            else:
                class_definition += f"        {str(symbols(f'x{abs(clause_output[i])}')) + ' = ' + replace_boolean_operators(str(clause_gate[i]))}\n"

    class_definition += f"        outputs = {', '.join(outputs+gate_constraint)}, \n"
    class_definition += f"        variables = {', '.join(variables)}\n"
    class_definition += "\n" \
                        f"        return outputs, variables\n"
    return class_definition, len(gate_constraint)

