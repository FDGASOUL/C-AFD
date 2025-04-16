# def closure(attrs, fds):
#     """
#     计算属性集 attrs 在 FDs 下的闭包。
#     attrs: 可迭代的属性（如 tuple/list/set）
#     fds: 可迭代的 FD，每个 FD 是 (lhs, rhs)，lhs 可迭代，rhs 单属性
#     返回一个 set，表示 attrs+。
#     """
#     closure_set = set(attrs)
#     changed = True
#     while changed:
#         changed = False
#         for lhs, rhs in fds:
#             if set(lhs).issubset(closure_set) and rhs not in closure_set:
#                 closure_set.add(rhs)
#                 changed = True
#     return closure_set
#
#
# def is_redundant(fd, all_fds):
#     """
#     判断 fd 是否在 all_fds 中冗余。
#     fd: (lhs, rhs)
#     all_fds: 整个依赖集合
#     """
#     lhs, rhs = fd
#     # 从全集中去掉这条 fd
#     reduced_fds = [f for f in all_fds if f != fd]
#     # 如果 rhs 已经在 X+ 中，就说明可以通过其他 FDs 推导，故冗余
#     return rhs in closure(lhs, reduced_fds)
#
#
# def filter_redundant_dependencies(fds):
#     """
#     从 fds 中移除所有冗余 FD，返回一个最小化后的集合。
#     """
#     minimal = []
#     for fd in fds:
#         # 注意：在判断冗余时，要用全集 fds，而不是仅用当前已选的 minimal
#         if not is_redundant(fd, fds):
#             minimal.append(fd)
#     return minimal
#
#
# all_discovered_dependencies = [(['column1'], 'column2'), (['column3'], 'column4'), (['column5', 'column6'], 'column7'), (['column8', 'column9'], 'column10'), (['column11'], 'column13'), (['column12'], 'column13'), (['column16'], 'column14'), (['column16'], 'column15')]
# minimal_fds = filter_redundant_dependencies(all_discovered_dependencies)
# print("Minimal FDs:", minimal_fds)


def is_redundant(fd, dependencies):
    """判断一条FD是否是冗余的（使用闭包计算）"""
    lhs, rhs = fd
    closure = set(lhs)
    # 通过迭代计算闭包
    changed = True
    while changed:
        changed = False
        for dep_lhs, dep_rhs in dependencies:
            if set(dep_lhs).issubset(closure) and dep_rhs not in closure:
                closure.add(dep_rhs)
                changed = True
    return rhs in closure


def filter_redundant_dependencies(dependencies):
    """筛选冗余的函数依赖（带排序优化）"""
    # 按左部长度升序排列，相同长度按右部字母排序
    sorted_deps = sorted(dependencies,
                         key=lambda x: (len(x[0]), x[1]))

    filtered_dependencies = []
    for fd in sorted_deps:
        if not is_redundant(fd, filtered_dependencies):
            filtered_dependencies.append(fd)
    return filtered_dependencies


all_discovered_dependencies = [(['ProviderNumber'], 'HospitalName'), (['ProviderNumber'], 'City'), (['ZIPCode'], 'City'), (['PhoneNumber'], 'City'), (['ProviderNumber'], 'State'), (['City'], 'State'), (['ZIPCode'], 'State'), (['PhoneNumber'], 'State'), (['ProviderNumber'], 'ZIPCode'), (['PhoneNumber'], 'ZIPCode'), (['ProviderNumber'], 'CountyName'), (['ProviderNumber'], 'PhoneNumber'), (['ProviderNumber'], 'HospitalOwner'), (['PhoneNumber'], 'HospitalOwner'), (['MeasureCode'], 'Condition'), (['MeasureName'], 'Condition'), (['StateAvg'], 'Condition'), (['MeasureName'], 'MeasureCode'), (['StateAvg'], 'MeasureCode'), (['MeasureCode'], 'MeasureName'), (['StateAvg'], 'MeasureName')]
minimal_fds = filter_redundant_dependencies(all_discovered_dependencies)
print("Minimal FDs:", minimal_fds)
print(len(minimal_fds))
