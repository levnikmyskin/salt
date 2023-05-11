from baselines.cormack_knee import KneeStopping


class BudgetStopping(KneeStopping):
    def __str__(self):
        return "BudgetKnee"

    def _check_rho(self, current_rho, rho, num_docs, current_rels, n_assessed):
        return (current_rho >= 6 and n_assessed >= 10 * num_docs / current_rels) or (
            n_assessed >= num_docs * 0.75
        )
