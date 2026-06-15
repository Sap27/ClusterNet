try:
    from gprofiler import GProfiler
    print("gprofiler OK")
except ImportError as e:
    print(f"gprofiler MISSING: {e}")

try:
    from goatools.obo_parser import GODag
    print("goatools OK")
except ImportError as e:
    print(f"goatools MISSING: {e}")

try:
    from scipy.stats import fisher_exact
    print("scipy.stats OK")
except ImportError as e:
    print(f"scipy MISSING: {e}")
