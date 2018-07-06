from pypsdd import Vtree, SddManager, PSddManager, SddNode, Inst, io
import sys

""" This is the main way in which SDDs should be used to compute semantic loss.
Construct an instance from a given SDD and vtree file, and then use the available
functions for computing the most probable explanation, weighted model count, or
constructing a tensorflow circuit for integrating semantic loss into a project.
"""


class CircuitMPE:
    def __init__(self, vtree_filename, sdd_filename):
        # Load the Sdd, convert to psdd
        vtree = Vtree.read(vtree_filename)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_filename, manager)
        pmanager = PSddManager(vtree)
        # Storing psdd
        self.beta = pmanager.copy_and_normalize_sdd(alpha, vtree)


    def compute_mpe_inst(self, lit_weights, binary_encoding=True):
        mpe_inst = self.beta.get_weighted_mpe(lit_weights)[1]
        print self.beta.model_count()
        if binary_encoding:
            # Sort by variable, but ignoring negatives
            mpe_inst.sort(key=lambda x: abs(x))
            return [int(x > 0) for x in mpe_inst]
        else:
            return mpe_inst

    def weighted_model_count(self, lit_weights):
        return self.beta.weighted_model_count(lit_weights)

    def get_tf_ac(self, litleaves):
        return self.beta.generate_tf_ac(litleaves)

if __name__ == '__main__':
    c = CircuitMPE(sys.argv[1], sys.argv[2])
    lit_weights = [[1, 0] for x in xrange(25)]
    # lit_weights[0] = [0.1, 0.9]
    # lit_weights[1] = [0.1, 0.9]
    lit_weights[0] = [0, 1]
    lit_weights[6] = [0, 1]
    lit_weights[12] = [0, 1]
    lit_weights[18] = [0, 1]
    lit_weights[24] = [0, 1]
    print c.weighted_model_count(lit_weights)
    print sorted(c.compute_mpe_inst(lit_weights, binary_encoding=False))
