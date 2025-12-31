# Guide-level dose response

This analysis links guide-level on-target expression measurements to guide-level distal differential-expression counts. Models are compared by held-out target pairs, with one guide used as an anchor to predict its paired guide. Target-state effects cancel within each pair.

Only downstream-gene counts are available at guide resolution. Donor-specific guide effects, signed response norms, module activity, rerouting, disease convergence, and target-specific nonlinear curves are retained as unavailable or underpowered outputs.
