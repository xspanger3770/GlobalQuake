package globalquake.core.earthquake;

public class EfficientMedianCalculator {
    public static long findMedian(long[] nums) {
        int n = nums.length;
        if (n % 2 == 0) {
            int k1 = n / 2;
            int k2 = k1 - 1;
            return (quickselect(nums, k1) + quickselect(nums, k2)) / 2;
        } else {
            int k = n / 2;
            return quickselect(nums, k);
        }
    }

    public static long quickselect(long[] nums, int k) {
        int low = 0, high = nums.length - 1;

        while (low <= high) {
            int pivotIndex = partition(nums, low, high);
            if (pivotIndex == k) {
                return nums[pivotIndex];
            } else if (pivotIndex < k) {
                low = pivotIndex + 1;
            } else {
                high = pivotIndex - 1;
            }
        }

        return -1; // Error case
    }

    public static int partition(long[] nums, int low, int high) {
        long pivot = nums[high];
        int i = low;

        for (int j = low; j < high; j++) {
            if (nums[j] <= pivot) {
                swap(nums, i, j);
                i++;
            }
        }

        swap(nums, i, high);
        return i;
    }

    public static void swap(long[] nums, int i, int j) {
        long temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
