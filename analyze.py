import csv

# Function to check if both minimum and second minimum values are between indexes 6 and 10
def is_min_and_second_min_between_6_and_10(arr):
    sorted_values = sorted(arr)
    min_value = sorted_values[0]
    second_min_value = sorted_values[1]
    min_index = arr.index(min_value)
    second_min_index = arr.index(second_min_value)
    return 6 <= min_index <= 10 and 6 <= second_min_index <= 10

# Function to check if values between indexes 1-5 and 11-15 are greater than min and second min
def are_values_greater_between_1_to_5_and_11_to_15(arr, min_value, second_min_value):
    return all(value > min_value and value > second_min_value for value in arr[:5] + arr[10:])

good_filename = '__4gaby-short-blink.txt'
bad_filename = '__4gaby-open-blink.txt'
# Read CSV file and analyze each line
with open('__gaby-openv4.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    with open(good_filename, 'w') as good_file, open(bad_filename, 'w') as bad_file:
        for row in csvreader:
            original_values = list(map(float, row))  # Convert values to float
            squared_values = [val ** 2 for val in original_values]  # Square each value for analysis
            sorted_values = sorted(squared_values)
            min_value = sorted_values[0]
            second_min_value = sorted_values[1]
            if is_min_and_second_min_between_6_and_10(squared_values) and are_values_greater_between_1_to_5_and_11_to_15(squared_values, min_value, second_min_value):
                good_file.write(' '.join(map(str, original_values)) + '\n')
            else:
                bad_file.write(' '.join(map(str, original_values)) + '\n')

print("Analysis complete. Results saved in good-results.txt and bad-results.txt.")
