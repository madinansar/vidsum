"""0/1 knapsack used to pick shots under the summary-length budget.

Identical to CA-SUM's evaluation/knapsack_implementation.py so F1 numbers stay comparable.
"""


def knapsack(capacity, weights, values, n=None):
    """Return the indices of the selected items (shots), maximizing total value within `capacity` frames."""
    n = len(weights) if n is None else n
    table = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                table[i][w] = 0
            elif weights[i - 1] <= w:
                table[i][w] = max(values[i - 1] + table[i - 1][w - weights[i - 1]], table[i - 1][w])
            else:
                table[i][w] = table[i - 1][w]

    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if table[i][w] != table[i - 1][w]:
            selected.insert(0, i - 1)
            w -= weights[i - 1]
    return selected
