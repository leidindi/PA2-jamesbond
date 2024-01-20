def calculate_combinations(total_length):
    combinations = []

    for window_size in range(1, total_length + 1):
        for overlap_percentage in range(1, 101):
            if (overlap_percentage * window_size) % 100 == 0:
                combinations.append((window_size, overlap_percentage))

    return combinations

def main():
    try:
        total_length = 3563

        result = calculate_combinations(total_length)

        print("Possible combinations of window size and percentage of overlap:")
        for combination in result:
            if combination[1]<50 and combination[0]< 250:
                print(f"Window Size: {combination[0]} = {combination[0]/200} seconds, Overlap Percentage: {combination[1]}%")

    except ValueError:
        print("Invalid input. Please enter a valid positive integer for total length.")

if __name__ == "__main__":
    main()
