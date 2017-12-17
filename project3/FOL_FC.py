#!/usr/bin/env python3
"""
CS 1571 Project 3: FOL Inference - Forward Chaining
Zac Yu (zhy46@)
"""

from argparse import ArgumentParser, FileType
from collections import namedtuple
from copy import deepcopy
from enum import Enum
from re import search


class AtomArgType(Enum):
    UNKNOWN = 0  # reserved
    CONSTANT = 1
    VARIABLE = 2


Atom = namedtuple('Atom', ['predicate_id', 'arguments'])
AtomArg = namedtuple('AtomArg', ['arg', 'type'])
Rule = namedtuple('Rule', ['lhs', 'rhs'])
FactSource = namedtuple('FactSource', ['rule_id', 'bindings'])


class FolFcSystem(object):
    """A first-order-logic system using forward chaining rules"""
    predicate_symbols = []
    constant_symbols = []
    rules = []
    facts = []
    fact_srcs = {}
    fc_counter = 0
    rf_counter = 0
    u_counter = 0

    @classmethod
    def is_fact(cls, atom):
        """Check is an atom is a fact"""
        for argument in atom.arguments:
            if argument.type != AtomArgType.CONSTANT:
                return False
        return True

    def add_fact(self, fact, source=False):
        """Add a fact to the knowledge base and invoke forward-chain"""
        if not fact or not self.is_fact(fact):
            return False
        if fact in self.facts:
            self.rf_counter += 1
            return True
        if source:
            self.fact_srcs[len(self.facts)] = source
        else:
            print('Fact', self.format_atom(fact))
        self.facts.append(fact)
        self.fc_counter += 1
        for rule_id, rule in enumerate(self.rules):
            self.find_and_infer(rule_id, rule.lhs, rule.rhs, {})
        return True

    def add_rule(self, rule):
        """Add a fact to the knowledge base"""
        if not rule:
            return False
        print('Rule', self.format_rule(rule))
        self.rules.append(rule)
        return True

    def ask(self, query):
        """Process a query and return a valid substitution if exists"""
        if not query or not self.is_fact(query):
            return False
        if query not in self.facts:
            return False
        fact_idx = self.facts.index(query)
        if fact_idx not in self.fact_srcs:
            print('From given,', self.format_atom(query))
            return True
        fact_src = self.fact_srcs[fact_idx]
        rule = self.rules[fact_src.rule_id]
        bindings = fact_src.bindings
        for premise in rule.lhs:
            self.ask(self.subst(bindings, premise))
        print('By rule', self.format_rule(rule), 'we have',
              self.format_atom(self.subst(bindings, rule.rhs)))
        return True

    def find_and_infer(self, rule_id, lhs, rhs, bindings):
        """Find feasible bindings and attempt firing inference rules"""
        if not lhs:
            self.add_fact(self.subst(bindings, rhs),
                          FactSource(rule_id, bindings))
            return
        for fact in self.facts:
            if lhs[0].predicate_id != fact.predicate_id:
                continue
            new_bindings = self.unify(self.subst(bindings, lhs[0]), fact)
            if new_bindings is not False:
                new_bindings.update(bindings)
                self.find_and_infer(rule_id, lhs[1:], rhs, new_bindings)

    def format_atom(self, fact):
        """Format an atom to a human readable string"""
        return (self.predicate_symbols[fact.predicate_id] + '(' +
                ', '.join(list(map(
                    lambda arg: (self.constant_symbols[arg.arg] if
                                 arg.type is AtomArgType.CONSTANT else
                                 'var' + str(arg.arg)),
                    fact.arguments))) + ')')

    def format_rule(self, rule):
        """Format a rule to a human radable string"""
        return (' ^ '.join(list(map(self.format_atom, rule.lhs))) + ' -> ' +
                self.format_atom(rule.rhs))

    def parse_atom(self, atom_str):
        """Parse an atom from string"""
        found = search(r'^(\w+)\(([\w,]+)\)$', atom_str)
        if not found:
            return False
        if found.group(1) not in self.predicate_symbols:
            pid = len(self.predicate_symbols)
            self.predicate_symbols.append(found.group(1))
        else:
            pid = self.predicate_symbols.index(found.group(1))
        arguments = found.group(2).split(',')
        for i in range(len(arguments)):
            if arguments[i] < 'a':
                if arguments[i] not in self.constant_symbols:
                    cid = len(self.constant_symbols)
                    self.constant_symbols.append(arguments[i])
                else:
                    cid = self.constant_symbols.index(arguments[i])
                arguments[i] = cid
                arg_type = AtomArgType.CONSTANT
            else:
                arg_type = AtomArgType.VARIABLE
            arguments[i] = AtomArg(arguments[i], arg_type)
        return Atom(pid, arguments)

    def parse_rule(self, rule_parts):
        """Parse a rule from expression parts"""
        lhs = []
        for i in range(len(rule_parts)):
            if i % 2 == 0:
                atom = self.parse_atom(rule_parts[i])
                if not atom:
                    return False
                if i < len(rule_parts) - 2:
                    lhs.append(atom)
                else:
                    rhs = atom
            elif rule_parts[i] != '^' and (
                    rule_parts[i] != '->' or i != len(rule_parts) - 2):
                return False
        # Normalize rule and variable names
        lhs.sort(key=lambda atom: atom.predicate_id)
        variable_symbols = []
        for i in range(len(lhs)):
            for j in range(len(lhs[i].arguments)):
                if lhs[i].arguments[j].type is AtomArgType.VARIABLE:
                    variable_name = lhs[i].arguments[j].arg
                    if variable_name not in variable_symbols:
                        vid = len(variable_symbols)
                        variable_symbols.append(variable_name)
                    else:
                        vid = variable_symbols.index(variable_name)
                    lhs[i].arguments[j] = AtomArg(vid, AtomArgType.VARIABLE)
        for i in range(len(rhs.arguments)):
            if rhs.arguments[i].type is AtomArgType.VARIABLE:
                rhs.arguments[i] = AtomArg(
                    variable_symbols.index(rhs.arguments[i].arg),
                    AtomArgType.VARIABLE)
        return Rule(lhs, rhs)

    def print_stats(self):
        """Get performance stats"""
        print('Activated forward-chain %s time(s)' % self.fc_counter)
        print('Attempted to add %s reduncent fact(s)' % self.rf_counter)
        print('Performed %s unification(s)' % self.u_counter)
        print('Registered %s predicate symbol(s)' %
              len(self.predicate_symbols))
        print('Registered %s constant symbol(s)' % len(self.constant_symbols))

    def subst(self, bindings, atom):
        """Substitute bindings into an atom"""
        result = deepcopy(atom)
        for i in range(len(atom.arguments)):
            if (result.arguments[i].type is AtomArgType.VARIABLE and
                    result.arguments[i].arg in bindings):
                result.arguments[i] = AtomArg(bindings[result.arguments[i].arg],
                                              AtomArgType.CONSTANT)
        return result

    def unify(self, atom, fact):
        """Unify an atom with a fact and return a list of bindings if exists"""
        self.u_counter += 1
        bindings = {}
        if (not self.is_fact(fact) or
                atom.predicate_id != fact.predicate_id or
                len(atom.arguments) != len(fact.arguments)):
            return False
        for i in range(len(atom.arguments)):
            if atom.arguments[i].type is AtomArgType.VARIABLE:
                if (atom.arguments[i].arg in bindings and
                        bindings[atom.arguments[i].arg] !=
                        fact.arguments[i].arg):
                    return False
                else:
                    bindings[atom.arguments[i].arg] = fact.arguments[i].arg
            elif atom.arguments[i].arg != fact.arguments[i].arg:
                return False
        return bindings


def main():
    """Main function"""
    parser = ArgumentParser(description='Solve FOL problem.')
    parser.add_argument('config_file', type=FileType('r'),
                        help='the configuration file for the problem')

    args = parser.parse_args()
    fol_fc = FolFcSystem()

    print('-' * 36 + ' Given ' + '-' * 36)
    for line in args.config_file:
        components = line.split()
        if len(components) > 2:
            if not fol_fc.add_rule(fol_fc.parse_rule(components)):
                raise SystemExit('error: invalid input line: ' + line)
        elif len(components) == 1:
            if not fol_fc.add_fact(fol_fc.parse_atom(components[0])):
                raise SystemExit('error: invalid input line: ' + line)
        else:
            if components[0] != 'PROVE':
                raise SystemExit('error: invalid input line: ' + line)
            query = fol_fc.parse_atom(components[1])
            print('-' * 36 + ' Promt ' + '-' * 36)
            print('Show that', fol_fc.format_atom(query))
            print('-' * 36 + ' Proof ' + '-' * 36)
            result = fol_fc.ask(query)
            if result is False:
                print('Failed')
            else:
                print('Q.E.D.')
            break
    print('-' * 36 + ' Stats ' + '-' * 36)
    fol_fc.print_stats()
    print('-' * 37 + ' Raw ' + '-' * 37)
    print(result)
    for fid, fact in enumerate(fol_fc.facts):
        if fid in fol_fc.fact_srcs:
            print(fol_fc.format_atom(fact))
        if fact == query:
            break


if __name__ == '__main__':
    main()
