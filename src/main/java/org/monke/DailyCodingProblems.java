package org.monke;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;


public class DailyCodingProblems {

    public DailyCodingProblems() { }

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Arrays, Lists, Binary Search.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Returns whether any two numbers from a list add up to k.
     *
     *
     * Examples :
     * - [10, 15, 3, 7] and k = 17, returns true since 10 + 7 = 17.
     *
     * Constraints :
     *  - 1 parse only.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * For each element, binary search the number k - element in the list.
     *
     * If found, a sum is present, return true.
     *
     * Else, go to the next element until the whole list have been parsed.
     *
     * Special case if the same index is found : since list is sorted,
     * should return false unless the same number is present twice : 2a = k.
     *
     * Previous and next element of the list are checked for this last possibility.
     *
     *
     *  Performances :
     * - Binary search is O(log n), so its O(n log n).
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def two_sum(lst, K):
     *     lst.sort()
     *
     *     for i in range(len(lst)):
     *         target = K - lst[i]
     *         j = binary_search(lst, target)
     *
     *         # Checks that binary search found the target and that it's not in the same index
     *         # as i. If it is in the same index, we can check lst[i + 1] and lst[i - 1] to see
     *         # if there's another number that's the same value as lst[i].
     *         if j == -1:
     *             continue
     *         elif j != i:
     *             return True
     *         elif j + 1 < len(lst) and lst[j + 1] == target:
     *             return True
     *         elif j - 1 >= 0 and lst[j - 1] == target:
     *             return True
     *     return False
     *
     * def binary_search(lst, target):
     *     lo = 0
     *     hi = len(lst)
     *     ind = bisect_left(lst, target, lo, hi)
     *
     *     if 0 <= ind < hi and lst[ind] == target:
     *         return ind
     *     return -1
     */

    /**
     * Returns whether any two numbers from a list add up to k.
     */
    public boolean twoSum(List<Integer> list, int k) {
        Collections.sort(list);

        for (int i = 0; i < list.size(); i++) {
            int target = k - list.get(i);
            int j = binarySearch(list, target);

            if (j > -1) {
                return j != i
                    || (j + 1 < list.size() && list.get(j + 1) == target)
                    || (j - 1 >= 0 && list.get(j - 1) == target); // Special conditions.
            }
        }
        return false;
    }

    /**
     * Finds a number in a sorted list using binary search.
     */
    public int binarySearch(List<Integer> list, int target) {
        int lowerBound = 0;
        int upperBound = list.size() - 1;

        while (lowerBound <= upperBound) {
            int mid = (lowerBound + upperBound) / 2;

            if (list.get(mid) == target) return mid;
            else if (list.get(mid) > target) upperBound = mid - 1;
            else lowerBound = mid + 1;
        }
        return -1;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Products, Partial Products, Arrays, Pointers.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given an array of integers return a new array,
     * such that each element at index i of the new array is the product of all the numbers in the original array,
     * except the one at index i.
     *
     *
     * Examples :
     * - [1, 2, 3, 4, 5] returns [120, 60, 40, 30, 24].
     * - [3, 2, 1] returns [2, 3, 6].
     *
     * Constraints :
     * - No division.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * The element at index i is the product of elements before and after its index.
     *
     * First, we generate a list of prefix products : each index is the product of all elements before (easy).
     * Then, we generate a list of suffix products : each index is the product of all elements after,
     * by parsing the list in reverse (easy).
     *
     * Finally, we parse the list and append the products from prefix and suffix to exclude the element at index i.
     *
     *
     * Performances :
     * - O(N) time and space, iterating over the input arrays = O(N) time,
     *   creating the prefix and suffix arrays take up O(N) space.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def products(nums):
     *     # Generates prefix products.
     *     prefix_products = []
     *     for num in nums:
     *         if prefix_products:
     *             prefix_products.append(prefix_products[-1] * num)
     *         else:
     *             prefix_products.append(num)
     *
     *     # Generates suffix products.
     *     suffix_products = []
     *     for num in reversed(nums):
     *         if suffix_products:
     *             suffix_products.append(suffix_products[-1] * num)
     *         else:
     *             suffix_products.append(num)
     *     suffix_products = list(reversed(suffix_products))
     *
     *     # Generates result.
     *     result = []
     *     for i in range(len(nums)):
     *         if i == 0:
     *             result.append(suffix_products[i + 1])
     *         elif i == len(nums) - 1:
     *             result.append(prefix_products[i - 1])
     *         else:
     *             result.append(prefix_products[i - 1] * suffix_products[i + 1])
     *     return result
     */

    /**
     * Returns an array containing the product of all elements at index i, excluding the element.
     */
    public List<Integer> exclusiveProducts(List<Integer> list) {

        // Generates the cumulative products with previous value from the start.
        List<Integer> prefixProducts = new ArrayList<>();

        for (int i = 0; i < list.size(); i++) {
            int number = list.get(i);

            if (prefixProducts.isEmpty()) {
                prefixProducts.add(number);
            }
            else {
                prefixProducts.add(prefixProducts.get(i - 1) * number);
            }
        }
        // Generates the cumulative products with previous value from the end.
        List<Integer> suffixProducts = new ArrayList<>();

        for (int i = list.size() - 1; i >= 0 ; i--) {
            int number = list.get(i);

            // Suffix index goes in the opposite direction.
            int j = list.size() - 1 - i;

            if (suffixProducts.isEmpty()) {
                suffixProducts.add(number);
            }
            else {
                suffixProducts.add(suffixProducts.get(j - 1) * number);
            }
        }
        Collections.reverse(suffixProducts);

        // Generates result.
        List<Integer> result = new ArrayList<>();

        for (int i = 0; i < list.size(); i++) {
            if (i == 0) {
                result.add(suffixProducts.get(i + 1));
            }
            else if (i == list.size() - 1) {
                result.add(prefixProducts.get(i - 1));
            }
            else {
                result.add(prefixProducts.get(i - 1) * suffixProducts.get(i + 1));
            }
        }
        return result;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Arrays, In-Place Swapping.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given an array of integers, find the lowest positive integer that does not exist in the array.
     * The array can contain duplicates and negative numbers as well.
     *
     * You can modify the input array in-place.
     *
     *
     * Examples :
     * - [3, 4, -1, 1] returns 2.
     * - [1, 2, 0] returns 3.
     *
     * Constraints :
     * - Linear complexity in time and constant in space.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * ---------------- 1 ---------------- *
     *
     * Cannot sort the array because no sorting algorithm runs in linear time.
     *
     * Best case : the array contains all numbers between 1 and array.size().
     * > The result is between 1 and array.size() + 1.
     *
     * If each element is within the size of the array, we swap it at the index equaling its value - 1 (start at 0),
     * until whole array is parsed, which sorts the array for elements in range only.
     * By the end of this process, all the positive numbers should be at an index equaling their value (-1 since indexes start at 0).
     *
     * Then a final parse will find the first element which not equals its index.
     *
     * Example :
     *      -> [3, 4, -1, 1]
     *      -> [-1, 4, 3, 1]
     *      -> [-1, 1, 3, 4]
     *      -> [1, -1, 3, 4]
     *
     *
     * Performances :
     * - O(N) in time, since we swap each element at most once.
     * - O(1) in space, we only use the original array.
     *
     * ---------------- 2 ---------------- *
     *
     * Another way to do this is by adding all the numbers to a set, and then use a counter initialized to 1,
     * then continuously increment the counter and check whether the value is in the set.
     *
     *
     * Performances :
     * - Much simpler, but runs in O(N) time and space, whereas the previous algorithm uses no extra space.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * ---------------- 1 ---------------- *
     *
     * def first_missing_positive(nums):
     *     if not nums:
     *         return 1
     *     for i, num in enumerate(nums):
     *         while i + 1 != nums[i] and 0 < nums[i] <= len(nums):
     *             v = nums[i]
     *             nums[i], nums[v - 1] = nums[v - 1], nums[i]
     *             if nums[i] == nums[v - 1]:
     *                 break
     *     for i, num in enumerate(nums, 1):
     *         if num != i:
     *             return i
     *     return len(nums) + 1
     *
     * ---------------- 2 ---------------- *
     *
     * def first_missing_positive(nums):
     *     s = set(nums)
     *     i = 1
     *     while i in s:
     *         i += 1
     *     return i
     */

    /**
     * Returns the lowest positive integer that does not exist in an array.
     */
    public int firstMissingPositive(List<Integer> list) {

        // Takes only positive values between 1 and size of array.
        for (int i = 0; i < list.size(); i++) {
            // Swaps until reaching corresponding index while staying within [1, size].
            while (list.get(i) != i + 1 && 0 < list.get(i) && list.get(i) <= list.size()) {

                int current = list.get(i);
                int position = list.get(current - 1); // Starts at 0;

                // Swaps by adding at position (switch right) and remove element (switch left).
                list.remove(current - 1);
                list.add(current - 1, current);
                list.remove(i);
                list.add(i, position);

                if (Objects.equals(list.get(i), list.get(current - 1))) {
                    break;
                }
            }
        }

        for (int i = 1; i <= list.size(); i++) {
            if (list.get(i - 1) != i) {
                return i;
            }
        }
        return list.size() + 1;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Closures, Scoped Functions, Lambdas, Functional Programming.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair.
     *
     * Given this implementation of cons:
     *
     * def cons(a, b):
     *     def pair(f):
     *         return f(a, b)
     *     return pair
     *
     * Implement car and cdr.
     *
     *
     * Examples :
     * - car(cons(3, 4)) returns 3.
     * - cdr(cons(3, 4)) returns 4.
     *
     * Constraints :
     *  - None.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * Closure :
     * - cons(a, b) returns a 'pair' function. So car(cons(a, b)) is car applied to the 'pair' function as parameter.
     * - The 'pair' function takes another lambda function f as parameter, and applies it to a and b.
     *
     * The abstract implementation of car is therefore to create this f function and pass it as param of pair, then return the first value.
     *
     * Source of car and cdr functions : https://en.wikipedia.org/wiki/CAR_and_CDR.
     *
     *
     *  Performances :
     * - N/A.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def car(pair):
     *     return pair(lambda a, b: a)
     *
     * def cdr(pair):
     *     return pair(lambda a, b: b)
     */

    /**
     * Translation of the <code>cons</code> function in Java.
     * <br>
     * <br>
     * Python code :
     * <br>
     * <br>
     * <pre>
     * def cons(a, b):
     *     def pair(f):
     *         return f(a, b)
     *     return pair
     * </pre>
     * @param a Generic T parameter.
     * @param b Generic U parameter.
     * @return A function defined through the traditional Function interface (1 argument, 1 return type),
     * which takes as argument another unknown function (lambda, BiFunction interface since having 2 arguments),
     * applies it and returns its R return type.
     */
    public <T, U, R> Function<BiFunction<T, U, R>, R> cons(T a, U b) {
        return f -> f.apply(a, b);
    }

    /**
     * Returns the first element of the tuple created by the cons pair function.
     */
    public <T,U> T car(Function<BiFunction<T,U,T>,T> cons) {
        return cons.apply((a,b) -> a);
    }

    /**
     * Returns the last element of the tuple created by the cons pair function.
     */
    public <T,U> U cdr(Function<BiFunction<T,U,U>,U> cons) {
        return cons.apply((a,b) -> b);
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * String Parsing, Combinations, Recursion vs Dynamic Programming.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the number of ways it can be decoded.
     *
     * You can assume that all messages can be decoded. For example, '001' is not allowed.
     *
     *
     * Examples :
     * - '111' returns 3, since it could be decoded as 'aaa', 'ka', and 'ak'.
     * - '101' returns 1, 'ja'.
     *
     * Constraints :
     *  - None.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * ---------------- 1 ---------------- *
     *
     * Combinations problem might be good for recursion.
     *
     * Since digits are between 1 and 26 :
     * - If the string is of length <= 1, there is only 1 encoding (empty string counts as 1).
     * - If the string starts with 0, no possibility.
     * - For a pair of numbers having their concatenation <= 26, we can have 2 letters or a single letter,
     *   so 1 encoding is added to the pool.
     * - For a pair of numbers having their concatenation > 26, we can only have 2 letters, no additional encoding.
     * - We can then call the function again on the remaining substring (minus 1 and 2 number according to the number of encodings we found).
     *
     * Example :
     * - '444' only has 1 encoding, so any substring ('44', '4') still has 1 encoding, no incrementation.
     *
     *
     * Performances :
     * - O(2N) in time, as every branch calls itself recursively at most twice.
     *
     * ---------------- 2 ---------------- *
     *
     * Build up the result from the end in the same way as solution 1, but without recursion (iteratively).
     *
     * Hold a map with keys being the indexes of each character in the string.
     * map.get(i) holds the number of ways to encode the substring s[i:].
     *
     * We then decrement i, and check if there is 1 or 2 possibilities with the substring s[i:i+2],
     * then increment the map by getting the value at index i+1 and adding 1 if there is an additional encoding with the concatenation.
     *
     * map.get(0) holds all possibilities at the end.
     *
     *
     * Performances :
     * - O(N) in time, each iteration takes O(1) for hashmap read and write operations.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * ---------------- 1 ---------------- *
     *
     * def num_encodings(s):
     *     if s.startswith('0'):
     *         return 0
     *     elif len(s) <= 1: # This covers empty string.
     *         return 1
     *
     *     total = 0
     *
     *     if int(s[:2]) <= 26:
     *         total += num_encodings(s[2:])
     *
     *     total += num_encodings(s[1:])
     *     return total
     *
     * ---------------- 2 ---------------- *
     *
     * def num_encodings(s):
     *     # On lookup, this hashmap returns a default value of 0 if the key doesn't exist.
     *     # cache[i] gives us the number of ways to encode the substring s[i:].
     *     cache = defaultdict(int)
     *     cache[len(s)] = 1 # Empty string is 1 valid encoding.
     *
     *     for i in reversed(range(len(s))):
     *         if s[i].startswith('0'):
     *             cache[i] = 0
     *         elif i == len(s) - 1:
     *             cache[i] = 1
     *         else:
     *             if int(s[i:i + 2]) <= 26:
     *                 cache[i] = cache[i + 2]
     *             cache[i] += cache[i + 1]
     *     return cache[0]
     */

    /**
     * Returns the number of combinations of letters mapped by their alphabet index in a string of numbers.
     */
    public int numEncodings(String s) {
        Map<Integer, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();

        for (int i = chars.length - 1; i >= 0; i--) {

            if (String.valueOf(chars[i]).equals("0")) {
                map.put(i, 0);
            }
            // First iteration, string of length 1, 1 possibility.
            else if (i == chars.length - 1) {
                map.put(i, 1);
            }
            else {
                if (Integer.parseInt("" + chars[i] + chars[i + 1]) <= 26) {
                    // Number holds 2 indexes, avoids duplicates by getting the number of encodings at i + 2.
                    map.put(i, map.getOrDefault(i + 2, 1));
                }
                // If different letters, the number of possibilities does not increase, simply copies the value at i + 1.
                map.put(i, map.getOrDefault(i, 0) + map.get(i + 1));
            }
        }
        // Empty string counts as one.
        return map.getOrDefault(0, 1);
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Combinations, Recursion vs Dynamic Programming.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * There exists a staircase with N steps, and you can climb up either 1 or 2 steps at a time.
     * Given N, write a function that returns the number of unique ways you can climb the staircase. 
     * 
     * The order of the steps matters.
     * 
     * Bonus, generalize the climbable steps to any number from a set of positive integers X (not just 1 or 2).
     *
     * 
     * Examples :
     * - N = 4 steps, there are 5 unique ways :
     *      -> [1, 1, 1, 1].
     *      -> [2, 1, 1].
     *      -> [1, 2, 1].
     *      -> [1, 1, 2].
     *      -> [2, 2].
     *
     * Constraints :
     *  - None.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * ---------------- 1 ---------------- *
     *
     * Let's find a pattern to generalize :
     *
     *      N = 1: [1].
     *      N = 2: [1, 1], [2].
     *      N = 3: [1, 2], [1, 1, 1], [2, 1].
     *      N = 4: [1, 1, 2], [2, 2], [1, 2, 1], [1, 1, 1, 1], [2, 1, 1].
     *     
     * The only ways to get to N = 3, is to first get to N = 1, and then go up by 2 steps, or get to N = 2 and go up by 1 step.
     * So f(3) = f(2) + f(1).
     * Similarly, f(4) = f(3) + f(2).
     * 
     * In a general way, to climb to the next step, we can come from the previous one or the one before :
     *      -> f(n) = f(n - 1) + f(n - 2) = Fibonacci Sequence.
     *
     * For any combination of steps in X :
     *
     *      -> f(n) = sum(f(n - x)), for x ∈ X.
     *
     *
     * Performances :
     * - O(|X|N) in space and time, |X| ( = len(X) ) recursion calls.
     *
     * ---------------- 2 ---------------- *
     *
     * Increase performances by using dynamic programming (iteratively).
     *
     * Hold a map with keys being the steps, and values the number of ways we can get to step key with the set X.
     * Then for each next step, increments by summing f(n - x).
     *
     *
     * Performances :
     * - O(|X|N) time and O(N) space.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * ---------------- 1 ---------------- *
     *
     * def staircase(n, X):
     *     if n < 0:
     *         return 0
     *     elif n == 0:
     *         return 1
     *     else:
     *         return sum(staircase(n - x, X) for x in X)
     *
     * ---------------- 2 ---------------- *
     *
     * # 2 intervals.
     * def staircase(n):
     *     a, b = 1, 2
     *     for _ in range(n - 1):
     *         a, b = b, a + b
     *     return a
     *
     * # X intervals.
     * def staircase(n, X):
     *     cache = [0 for _ in range(n + 1)]
     *     cache[0] = 1
     *     for i in range(1, n + 1):
     *         cache[i] += sum(cache[i - x] for x in X if i - x >= 0)
     *     return cache[n]
     */

    /**
     * Returns the number of unique ways to climb a staircase of N steps,
     * by moving steps in different intervals provided in a list.
     */
    public int staircase(int steps, List<Integer> intervals) {
        Map<Integer, Integer> map = new HashMap<>();

        map.put(0, 1);

        for (int i = 1; i <= steps; i++) {
            // Summing f(n - x).
            for (int interval : intervals) {
                if (i >= interval) { // Interval of 5 steps in a 4 steps staircase for instance.
                    // Holds the number of ways to get to step i with the intervals.
                    map.put(i, map.getOrDefault(i, 0) + map.get(i - interval));
                }
            }
        }
        return map.get(steps);
    }

    // ------------------------------------------------------------------------------------------- //

    // TODO
    /**
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * String Parsing, Running Window.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given an integer k and a string s, find the length of the longest substring that contains at most k distinct characters.
     * 
     *
     * Examples :
     * - s = 'abcba' and k = 2 returns 'bcb'.
     *
     * Constraints :
     *  - None.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * Brute force solution : 
     * - Try every possible substring of the string and check whether it contains at most k distinct characters.
     * - If so and greater than the current valid substring, update.
     * 
     * Poor performances : O(n^2 * k) time, n^2 for every possible substring, and k for each character.
     *
     * Improvement : keep a running window of the longest substring.
     * A map hold characters to the index of their last occurrence.
     *
     * During each iteration, check the size of the window :
     * - If > k, pop the smallest item and recompute the bounds.
     *
     * Example :
     * - map = [a, b, c, b, a] bounds = (0, 0).
     *
     *      -> [-1, 4, 3, 1]
     *      -> [-1, 1, 3, 4]
     *      -> [1, -1, 3, 4]
     *
     *
     * We can improve this by instead keeping a running window of our longest substring. We'll keep a dictionary that maps characters to the index of their last occurrence.
     * Then, as we iterate over the string, we'll check the size of the dictionary.
     * If it's larger than k, then it means our window is too big, so we have to pop the smallest item in the dictionary and recompute the bounds.
     * If, when we add a character to the dictionary, and it doesn't go over k, then we're safe -- the dictionary hasn't been filled up yet, or it's a character we've seen before.
     *
     *
     * Performances :
     * - O(n * k) time and O(k) space.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def longest_substring_with_k_distinct_characters(s, k):
     *     if k == 0:
     *         return 0
     *
     *     # Keeps a running window.
     *     bounds = (0, 0)
     *     h = {}
     *     max_length = 0
     *     for i, char in enumerate(s):
     *         h[char] = i
     *         if len(h) <= k:
     *             new_lower_bound = bounds[0] # lower bound remains the same
     *         else:
     *             # otherwise, pop last occurring char
     *             key_to_pop = min(h, key=h.get)
     *             new_lower_bound = h.pop(key_to_pop) + 1
     *
     *         bounds = (new_lower_bound, bounds[1] + 1)
     *         max_length = max(max_length, bounds[1] - bounds[0])
     *
     *     return max_length
     */

    /**
     *
     */
    public int longest_substring_with_k_distinct_characters(String s, int k) {
        if (k == 0) return 0;

        int[] bounds = new int[]{0, 0};
        Map<String, Integer> map = new HashMap<>();

        int result = 0;

        char[] chars = s.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            map.put(String.valueOf(chars[i]), i);

            // pop last occurring char
            if (map.size() > k) {

            }
            // else, character we've seen before
        }


        return 0;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Reservoir Sampling.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given a stream of elements too large to store in memory, pick a random element from the stream with uniform probability.
     *
     *
     * Examples :
     * - N/A.
     *
     * Constraints :
     *  - Constant complexity in space : cannot store values in memory.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * Cannot store elements in the list to pick a random one from [0, size - 1].
     *
     * The technique is to find a loop invariant :
     *
     * Let's assume we picked an element k uniformly from [0, i - 1], with a probability of 1 / i.
     * In order to maintain the loop invariant, we would need to pick this element k as the new random element at 1 / (i + 1) probability.
     *
     * We need to prove that at each step, the selected element has the same probability to be selected as in the steps before (invariant),
     * so that in the end, any element has an equal probability to be selected at any step.
     *
     * For i = 0 (first increment), the random element is the first one : 1 / (i + 1) = 1.
     * For i > 0 :
     *
     *      -> P(k is selected at i + 1) = P(k is not selected at i + 1) ∩ P(k is selected at i).
     *
     * The element has the probability 1 - 1 / (i + 1) to NOT be selected at step i + 1.
     * Since we assumed uniformity at step i :
     *
     *      -> P(k is selected at i + 1) = 1 - 1 / (i + 1) * 1 / i = 1 / (i + 1).
     *
     * Now every element of the stream has an equal chance to be selected by checking the value of a random number in [0, i] :
     * The method will return a number with a probability of 1 / i which is our invariant.
     * We check if this number is equal to 1 for instance and return the result.
     *
     * This technique is called Reservoir Sampling and can be extended to selecting k elements uniformly from a stream of size n :
     * https://en.wikipedia.org/wiki/Reservoir_sampling, https://balaramdb.com/2020/06/reservoir-sampling/
     *
     *
     * Performances :
     * - O(N) in time, since a single loop over the input elements is used.
     * - O(1) in space, only 1 variable is stored !
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * import random
     *
     * def reservoir_sampling(big_stream):
     *     random_element = None
     *
     *     for i, e in enumerate(big_stream):
     *         if random.randint(1, i + 1) == 1:
     *             random_element = e
     *     return random_element
     */

    /**
     * Returns a random element from an input stream with uniform probability.
     */
    public <T> T reservoirSampling(List<T> stream) {
        // Entry condition : i = 0.
        T result = stream.get(0);

        for (int i = 1; i < stream.size(); i++) {
            // Checks if the invariant equals to an arbitrary value, select and return.
            if ((int) (Math.random() * i) == 1) {
                result = stream.get(i);
            }
        }
        return result;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Arrays, Circular Buffer.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Implement a data structure to record the last N entries ids in a log, with the following API :
     *
     * - record(entry_id) : adds the entry_id to the log.
     * - get_last(i) : gets the ith last element from the log. i is guaranteed to be smaller than or equal to N.
     *
     *
     * Examples :
     * - N/A.
     *
     * Constraints :
     *  - As efficient in time and space as possible.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * An array is perfect for this problem. We can just initialize the array to have size N, and index it in constant time.
     * We can record an entry by adding it to the list. The records array must be FIFO.
     *
     * When the array is full, we have to pop the older element and add the new element at the end of the array.
     * This will require to shift all elements by 1 to the left, which takes 0(N) time.
     *
     * To solve this, store the index of the last added entry.
     * To get the last ith entry, simple take current - i the element.
     *
     * When the array is full, the index will be set at 0 and the first element will be replaced, preserving the FIFO aspect of the array.
     *
     *
     * Performances :
     * - 0(1) in space for the array, and 0(1) in time to write and read the array.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * class Log(object):
     *     def __init__(self, n):
     *         self.n = n
     *         self._log = []
     *         self._cur = 0
     *
     *     def record(self, entry_id):
     *         if len(self._log) == self.n:
     *             self._log[self._cur] = entry_id
     *         else:
     *             self._log.append(entry_id)
     *         self._cur = (self._cur + 1) % self.n
     *
     *     def get_last(self, i):
     *         return self._log[self._cur - i]
     */

    /**
     * Implements a data structure to record the last N entries ids in a log.
     */
    public static class Log<T> {

        int recordSize;
        int current = 0;
        List<T> record = new ArrayList<>();

        public Log(int recordSize) { this.recordSize = recordSize; }

        /** Inserts an entry in the record. */
        public void record(T entry) {
            // current would have been already set at 0, circling back to the first element.
            if (record.size() == recordSize) {
                record.remove(current);
                record.add(current, entry);
            } else {
                // No problem, appends at the end.
                record.add(entry);
            }
            // Circles to 0 when reaching the record length.
            current = (current + 1) % recordSize;
        }

        /** Returns the last ith element of the log from the record. */
        public T getLast(int i) {
            int index = current - i;

            if (current < i) {
                index = recordSize - (i - current);
            }
            return record.get(index);
        }
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Arrays, Double-Ended Queues.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given an array of integers of size n, and a number k in [1, n],
     * compute the maximum values of each subarray of consecutive values of length k.
     *
     * You can modify the input array in-place, and you do not need to store the results.
     * You can simply print them out as you compute them.
     *
     *
     * Examples :
     * - [10, 5, 2, 7, 8, 7] and k = 3 returns [10, 7, 8, 8], since :
     *
     *     10 = max(10, 5, 2)
     *     7 = max(5, 2, 7)
     *     8 = max(2, 7, 8)
     *     8 = max(7, 8, 7)
     *
     * Constraints :
     *  - O(n) time and O(k) space.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * Brute force solution runs in O(n * k) time, by taking all sub-arrays for each index : max(array[i:i + k]).
     *
     * Better solution :
     * - Use a max-heap of size k and add the first k elements to the heap initially,
     * and then pop off the max and add the next element for the rest of the array.
     * - Adding and extracting from the heap takes O(log k), so this algorithm will take O(n * log k).
     *
     * Trick :
     * - For [1, 2, 3, 4, 5, 6, 7, 8, 9] and k = 3, after evaluating the max of first range, since 3 is at the end,
     *   we only need to check whether 4 is greater than 3. If it is, then we can print 4 immediately, and if it isn't, we can stick with 3.
     *
     * - For [9, 8, 7, 6, 5, 4, 3, 2, 1] and k = 3, after evaluating the max of the first range, we can't do the same thing, since we can't use 9 again.
     *   We have to look at 8 instead, and then once we move on to the next range, we have to look at 7.
     *
     * Since we're restrained in space to a structure of size k, we can use a queue q of size k to store indexes.
     * We use a loop invariant : at each iteration, q is a list of indices where their corresponding values are in descending order.
     *
     * To do so, we use a double ended queue.
     *
     * We parse the array. Each iteration gives us a new subarray array[i - k:i]. We check the element, if it's greater than the first element of the dequeue
     * (the max, since the queue is ordered), we add it from the front and pop the first element.
     * Else, we enqueue it from the back to keep the invariant.
     *
     * At first, we parse the first k elements only to fill up the queue without popping.
     *
     * We then store or print the value of the front index of the queue, which is the max for the subarray.
     *
     * Example :
     * - [10, 5, 2, 7, 8, 7] and k = 3.
     *
     *      -> Preprocessing.
     *
     *          After processing 10: [0].
     *          After processing 5: [0, 1] # 5 is smaller than 10, and 10 is still valid until we hit the 3rd index.
     *          After processing 2: [0, 1, 2] # 2 is smaller than 5, and 10 is still valid.
     *
     *      -> Main Loop.
     *
     *          Print 10.
     *          After processing 7: [4] # 10 is no longer valid (we can tell since the current index - 0 > k), so we dequeue from the front.
     *          7 is bigger than 5 and 2, so we get rid of them from the back and replace it with the 7.
     *          Print 7.
     *          After processing 8: [5] # 8 is bigger than 7, get rid of it from the back and replace it with 8.
     *          Print 8.
     *          After processing 7: [5, 4] # 7 is smaller than 8, so we enqueue it from the back.
     *          Print 8.
     *
     *
     * Performances :
     * - O(n) time (1 parse on the array) and O(k) space (the queue).
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def max_of_subarrays(lst, k):
     *     q = deque()
     *     for i in range(k):
     *         while q and lst[i] >= lst[q[-1]]:
     *             q.pop()
     *         q.append(i)
     *
     *     # Loop invariant: q is a list of indices where their corresponding values are in descending order.
     *     for i in range(k, len(lst)):
     *         print(lst[q[0]])
     *         while q and q[0] <= i - k:
     *             q.popleft()
     *         while q and lst[i] >= lst[q[-1]]:
     *             q.pop()
     *         q.append(i)
     *     print(lst[q[0]])
     */

    /**
     * Returns a list of the maximum values of each subarray of consecutive values of length k.
     */
    public List<Integer> maxOfSubArrays(List<Integer> list, int k) {
        // Stores and returns results for testing purposes, but should print instead.
        List<Integer> maxes = new ArrayList<>();
        Deque<Integer> q = new ArrayDeque<>(k);

        // Preprocessing : gets the max of 3 first indices.
        for (int i = 0; i < k; i++) {
            while (!q.isEmpty() && list.get(i) >= list.get(q.getLast())) {
                q.removeFirst();
            }
            q.addLast(i);
        }

        // Remaining array.
        for (int i = k; i < list.size(); i++) {
            maxes.add(list.get(q.getFirst()));

            // Removes indexes for elements out of the range of the subarray.
            while (!q.isEmpty() && q.getFirst() <= i - k) {
                q.removeLast();
            }
            // Adds index of greater element to the front and remove useless lower elements indexes.
            while (!q.isEmpty() && list.get(i) >= list.get(q.getLast())) {
                q.removeFirst();
            }
            q.addLast(i);
        }
        // Last iteration.
        maxes.add(list.get(q.getFirst()));

        return maxes;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Linked Lists, Intersections, Pointers.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given two linked lists of size M and N that intersect at some point, find the intersecting node.
     * The lists are non-cyclical.
     *
     *
     * Examples :
     * - A = 3 -> 7 -> 8 -> 10 -> 15 -> 6 and B = 99 -> 1 -> 8 -> 10, returns 8.
     *
     * Constraints :
     *  - O(M + N) time and constant space.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * Brute force : for each value of list 1, parse list 2 : O(N * M) in time.
     *
     * We can't use a map to store values as it would then be O(max(N, M)) in space.
     *
     * Trick :
     * - Get two pointers at the head of each list.
     * - Move the pointer of the larger list up by the difference of length of both lists,
     *   and then move the pointers forward in conjunction and check if they match (have to cycle the smaller list more than one time).
     *
     * This will parse all elements of both lists even when they are of different length.
     *
     * Example :
     * - A = 3 -> 7 -> 8 -> 10 -> 15 -> 6
     * - B = 99 -> 1 -> 8 -> 10
     *
     *      -> i : 8  | i : 10 | i : 15 | i : 6  | i : 3  | i : 7 | i : 8
     *      -> j : 99 | j : 1  | j : 8  | j : 10 | j : 99 | j : 1 | j : 8
     *
     *
     * Performances :
     * - Constant space (2 pointers) and O(M + N + C) in time.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def length(head):
     *     if not head:
     *         return 0
     *     return 1 + length(head.next)
     *
     * def intersection(a, b):
     *     m, n = length(a), length(b)
     *     cur_a, cur_b = a, b
     *
     *     if m > n:
     *         for _ in range(m - n):
     *             cur_a = cur_a.next
     *     else:
     *         for _ in range(n - m):
     *             cur_b = cur_b.next
     *
     *     while cur_a != cur_b:
     *         cur_a = cur_a.next
     *         cur_b = cur_b.next
     *     return cur_a
     */

    // TODO secure if no intersection is present.
    /**
     * Returns the intersecting node of 2 linked list, if present, else returns null.
     */
    public <T> T intersection(LinkedList<T> lList1, LinkedList<T> lList2) {
        int m = lList1.size();
        int n = lList2.size();

        Iterator<T> it1 = lList1.iterator();
        Iterator<T> it2 = lList2.iterator();

        T i = it1.next();
        T j = it2.next();

        if (m > n) {
            for (int h = 0; h < m - n; h++) {
                i = it1.next();
            }
        }
        else {
            for (int h = 0; h < n - m; h++) {
                j = it2.next();
            }
        }

        while (!i.equals(j)) {
            i = it1.next();
            j = it2.next();
        }
        return i;
    }

    // ------------------------------------------------------------------------------------------- //

    /*
     * ------------------------------------------------------------------------------------------- *
     *                                           Keywords
     * ------------------------------------------------------------------------------------------- *
     *
     * Arrays, Pointers.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                           Problem
     * ------------------------------------------------------------------------------------------- *
     *
     * Given an array of time intervals (start, end) for classroom lectures (possibly overlapping),
     * find the minimum number of rooms required.
     *
     *
     * Examples :
     * - [(30, 75), (0, 50), (60, 150)] returns 2.
     *
     * Constraints :
     *  - None.
     *
     * ------------------------------------------------------------------------------------------- *
     *                                          Resolution
     * ------------------------------------------------------------------------------------------- *
     *
     * Key observation :
     * - If no intervals overlap, there is no need for more than 1 room.
     * - The minimum number of rooms is the maximum number of overlapping intervals.
     *
     * Brute force would be to check every other interval in O(n^2).
     *
     * One solution is to extract the start times and end times of all the intervals and sort them.
     * Then we can start two pointers on each list, and consider the following :
     *
     * If the current start is before the current end, then we have a new overlap. Increment the start pointer.
     * If the current start is after the current end, then our overlap closes. Increment the end pointer.
     *
     * All that's left to do is keep a couple variables to keep track of the maximum number of overlaps we've seen so far and the current number of overlaps.
     *
     *
     * Performances :
     * - O(n log n) time for parsing and sorting intervals.
     *
     * ------------------------------------------- Code ------------------------------------------ *
     *
     * def max_overlapping(intervals):
     *      starts = sorted(start for start, end in intervals)
     *      ends = sorted(end for start, end in intervals)
     *
     *      current_max = 0
     *      current_overlap = 0
     *      i, j = 0, 0
     *      while i < len(intervals) and j < len(intervals):
     *          if starts[i] < ends[j]:
     *              current_overlap += 1
     *              current_max = max(current_max, current_overlap)
     *              i += 1
     *          else:
     *              current_overlap -= 1
     *              j += 1
     *      return current_max
     */

    /**
     * Returns the maximum number of overlaps in an array of time intervals [start, end].
     * 1 overlap is the default value.
     */
    public int maxOverlapping(List<int[]> intervals) {
        List<Integer> starts = intervals.stream()
            .map(interval -> interval[0])
            .sorted().toList();

        List<Integer> ends = intervals.stream()
            .map(interval -> interval[1])
            .sorted().toList();

        int i = 0;
        int j = 0;
        int maxOverlaps = 0;
        int currentOverlaps = 0;

        // Parses intervals.
        while (i < intervals.size() && j < intervals.size()) {
            // A new overlap is added to the list of active ones.
            if (starts.get(i) < ends.get(j)) {
                currentOverlaps++;
                maxOverlaps = Math.max(maxOverlaps, currentOverlaps);
                i++;
            }
            // The overlap has been resolved.
            else {
                currentOverlaps--;
                j++;
            }
        }
        return maxOverlaps;
    }

    // ------------------------------------------------------------------------------------------- //
}