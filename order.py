from clark import calc_route
from fields import ITEM_FIELDS

class Order:
  def __init__(self, items, savings):
    self.items = items
    self.route = calc_route(savings, [i[ITEM_FIELDS.ITEM_ID] for i in self.items])
    self.weight = sum([i[ITEM_FIELDS.WEIGHT] for i in self.items])

global_order_list = list[Order]() # orders = []
