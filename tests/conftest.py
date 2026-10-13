"""Suite-wide setup."""

import matplotlib

# One headless backend for every test, set before any test module imports pyplot.
matplotlib.use("Agg")
