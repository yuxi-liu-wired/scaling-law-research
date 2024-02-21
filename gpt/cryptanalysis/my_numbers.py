import numpy
from nistrng import *


def nist_test_suite(binary_sequence):
    print(binary_sequence)
    eligible_battery: dict = check_eligibility_all_battery(
        binary_sequence, SP800_22R1A_BATTERY
    )
    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
    # Test the sequence on the eligible tests
    results = run_all_battery(binary_sequence, eligible_battery, False)
    # Print results one by one
    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print(
                "- PASSED - score: "
                + str(numpy.round(result.score, 3))
                + " - "
                + result.name
                + " - elapsed time: "
                + str(elapsed_time)
                + " ms"
            )
        else:
            print(
                "- FAILED - score: "
                + str(numpy.round(result.score, 3))
                + " - "
                + result.name
                + " - elapsed time: "
                + str(elapsed_time)
                + " ms"
            )


save_file = "wiki_arithmetic_code_gpt2-xl_test_1"
with open(f"{save_file}.txt", "r", encoding="utf8") as f:
    code = f.read()
binary_sequence = numpy.array([int(i) for i in code], dtype=int)

nist_test_suite(binary_sequence)
