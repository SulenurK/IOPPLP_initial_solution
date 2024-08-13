def calc_route(savings, order_items, routes=None):
  if not routes:
    if len(order_items) == 1:
      return [int(order_items[0])] # cast to int from numpy int, cosmetic
    else:
      routes = []
  for s, i, j in savings:
    # print("routes:", routes)
    # print("Saving: ", s, i, j)
    if i not in order_items or j not in order_items:
      continue
    route_i = next((route for route in routes if i in route), None)
    route_j = next((route for route in routes if j in route), None)

    # print("route_i :", route_i, " route_j: ", route_j)
    # 1. No exisitng route has any of the nodes. Create a new route. Add the saving to new route. CONTINUE.
    if route_i is None and route_j is None:
      routes.append([i, j])
      continue
    # 2. Both nodes are there in the existing routes:
    elif route_i is not None and route_j is not None:
      # 2.1. Both nodes are in the same route. CONTINUE.
      if route_i == route_j:
        continue
      # 2.2. Both nodes are in the different routes:
      else:
        # 2.2.1. Both nodes are NOT in the interior. Merge both routes. CONTINUE. 
        if (route_i[0] == i or route_i[-1] == i) and (route_j[0] == j or route_j[-1] == j):
          # routes.remove(route_i)
          routes.remove(route_j)
          ## i sona, j basa gelecek
          if route_i[0] == i:
            route_i.reverse()
          if route_j[0] != j:
            route_j.reverse()
          route_i.extend(route_j)
        else:
          continue
    # 3. One exisitng route has one of the nodes:
    else:
      # 3.1. The node is in the interior of the exisitng route. CONTINUE.
      # 3.2. The node is NOT in the interior of the existing route:
      #     3.2.1. Node is at the beginning of the route. Add the other node at the beginning of that route. CONTINUE.
      #     3.2.2. Node is at the end of the route. Add the other node at the end of that route. CONTINUE.
      if route_i is not None:
        if route_i[0] == i:
          route_i.reverse()
          route_i.append(j)
        elif route_i[-1] == i:
          route_i.append(j)
      else: # route_j is not None
        if route_j[0] == j:
          route_j.reverse()
          route_j.append(i)
        elif route_j[-1] == j:
          route_j.append(i)
  return routes[0]

