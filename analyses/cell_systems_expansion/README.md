# Extended validation freeze

`freeze.py` records the preregistered hypotheses, gates, deterministic splits, every existing analysis configuration, prior freeze manifests, and the reused processed CD4 and validation artifacts. It must run before any new external outcome is opened. Existing K562 and RPE1 outcomes are marked as previously inspected, and RPE1 is not an untouched holdout. Large raw source objects remain covered by the data manifest and upstream audit; the expansion directly hashes the validated processed matrices it reuses.

Later blocks may stop at availability or power gates. They may not change this configuration after viewing outcomes. Any necessary correction requires a dated machine-readable amendment that states its scope and whether any outcome was visible.
