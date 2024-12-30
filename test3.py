# if len(dirty_rows_in_merge) == 1 or (len(dirty_rows_in_merge) == 2 and row_dirty_count[idx1] == 0):
#     if (r1 in dirty_rows_in_merge and row_dirty_count[r2] == 0) or (
#             r2 in dirty_rows_in_merge and row_dirty_count[r1] == 0):
#         continue
#     elif (r1 in dirty_rows_in_merge and row_dirty_count[r2] != 0) or (
#             r2 in dirty_rows_in_merge and row_dirty_count[r1] != 0):
#         current_benefit = max(row_dirty_count[r1], row_dirty_count[r2])
#         heapq.heappush(updated_heap, (-current_benefit, current_type, r1, r2))
#     else:
#         heapq.heappush(updated_heap, (-current_benefit, current_type, r1, r2))
# elif len(dirty_rows_in_merge) == 2 and row_dirty_count[idx1] != 0:
#     if (r1 in dirty_rows_in_merge and row_dirty_count[r2] == 0) or (
#             r2 in dirty_rows_in_merge and row_dirty_count[r1] == 0):
#         current_benefit = max(row_dirty_count[r1], row_dirty_count[r2])
#         heapq.heappush(updated_heap, (-current_benefit, current_type, r1, r2))
#     elif (r1 in dirty_rows_in_merge and row_dirty_count[r2] != 0) or (
#             r2 in dirty_rows_in_merge and row_dirty_count[r1] != 0):
#         current_benefit = Incorporate._calculate_row_benefit(expected_table[r1], expected_table[r2])
#         heapq.heappush(updated_heap, (-current_benefit, current_type, r1, r2))
#     else:
#         heapq.heappush(updated_heap, (-current_benefit, current_type, r1, r2))