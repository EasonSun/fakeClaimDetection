import heapq

class topK(object):
    # @param {int} k an integer
    def __init__(self, k):
        self.k = k
        self.nums = []
        heapq.heapify(self.nums)
        
    # @param {int} num an integer
    def add(self, nums):
        for num in nums:
            if len(self.nums) < self.k:
                heapq.heappush(self.nums, num)
            elif num > self.nums[0]:
                heapq.heappop(self.nums)
                heapq.heappush(self.nums, num)

    # @return {int[]} the top k largest numbers in array
    def topk(self):
        return sorted(self.nums, reverse=True)

tk = topK(5)
tk.add([5,7,7,9,3,5,1,3,8,4])
print tk.topk()