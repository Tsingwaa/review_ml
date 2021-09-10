def LIS(nums):
    dp = dict()

    for i in range(len(nums)):
        dp[i] = 1
        for j in range(i):
            if nums[j] < nums[i]:  # 比当前元素小的元素的元素
                dp[i] = max(dp[j] + 1, dp[i])

    return sorted(dp.values())[-1]


def max_subarray(nums):
    n = len(nums)
    if n == 0:
        return 0
    # define dp dict
    # dp = dict()  # 以i为结尾的最大子数组和
    #
    # # base case
    # dp[0] = nums[0]
    #
    # for i in range(1, n):
    #     dp[i] = max(nums[i], nums[i] + dp[i - 1])

    # return sorted(dp.values())[-1]

    last_maxsum = nums[0]
    max_sum = nums[0]

    for i in range(1, n):
        last_maxsum = max(nums[i], nums[i] + last_maxsum)
        if last_maxsum > max_sum:
            max_sum = last_maxsum

    return max_sum
