# LeetCode算法练习

全文按照《代码随想录》的题目顺序进行练习。

## 数组

### 二分查找

链接：https://leetcode.com/problems/binary-search/

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;

        while (left <= right) {
            int mid = left + ((right - left) >> 1); // 防止溢出，等效于(left + right) / 2
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                return mid;
            }
        }

        return -1;
    }
};
```

### 移除元素

链接：https://leetcode.com/problems/remove-element/description/

双指针法

```c++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow = 0, fast = 0;
        
        while (fast < nums.size()) {
            if (nums[fast] != val) {
                nums[slow++] = nums[fast++];
            } else {
                fast++;
            }
        }

        return slow;
    }
};
```

### 最短子数组

链接：https://leetcode.com/problems/minimum-size-subarray-sum/

使用滑动窗口算法

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int left = 0, res = INT_MAX, s = 0;
        
        for (int right = 0; right < nums.size(); right++) {
            s += nums[right];
            while (s >= target) {
                int len = right - left + 1;
                res = len < res ? len : res;
                s -= nums[left++];
            }
        }

        if (res == INT_MAX) return 0;

        return res;
    }
};
```

### 螺旋矩阵

链接：https://leetcode.com/problems/spiral-matrix-ii/

纯粹的模拟题

```c++
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        int x = 0, i = 0, j = 0;
        vector<vector<int>> res(n, vector<int>(n, 0));

        while (x < n * n) {
            while (j < n && res[i][j] == 0) {
                res[i][j++] = ++x;
            }
            i = i + 1, j = j - 1;

            while (i < n && res[i][j] == 0) {
                res[i++][j] = ++x;
            }
            i = i - 1, j = j - 1;

            while (j >= 0 && res[i][j] == 0) {
                res[i][j--] = ++x;
            }
            i = i - 1, j = j + 1;

            while (i >= 0 && res[i][j] == 0) {
                res[i--][j] = ++x;
            }
            i = i + 1, j = j + 1;
        }

        return res;
    }
};
```

