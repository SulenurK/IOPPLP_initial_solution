import numpy as np
from fields import ITEM_FIELDS
from order import global_order_list


item_orientation_vector = []

def init_item_orientation(length):
   global item_orientation_vector
   item_orientation_vector = np.zeros(length)

class Batch():
  num_batches = 0
  max_weight = 1500

  def __init__(self, V=(1200, 1000, 1500)):
    self.id = Batch.num_batches
    Batch.num_batches = Batch.num_batches + 1

    self.dimensions = V
    self.EMSs = [[np.array((0,0,0)), np.array(V)]]
    self.orders = []
    self.is_order_reversed = []

  # def set_max_weight(max_weight):
  #    Batch.max_weight = max_weight

  def __repr__(self):
    return f"Batch[{self.id}]: Orders:{self.orders}"

  def reset_num_batches():
     Batch.num_batches = 0

  def assign(self, items_in_order):

    item_order_id = items_in_order[0][ITEM_FIELDS.ORDER_ID]
    batch_weight = sum([global_order_list[order_id].weight for order_id in self.orders])
    order_weight = global_order_list[item_order_id].weight
    if batch_weight + order_weight > Batch.max_weight:
       return False

    # Take a backup of the state
    EMSs_bak = self.EMSs.copy()
    item_orientation_vector_bak = item_orientation_vector.copy()

    for item in items_in_order:
      selected_EMS, selected_orientation = self.DFTRC_2(item[ITEM_FIELDS.WIDTH:ITEM_FIELDS.HEIGHT+1])
      if selected_EMS == None:
        # Revert the state
        self.EMSs = EMSs_bak
        item_orientation_vector[:] = item_orientation_vector_bak
        return False

      self.update(selected_EMS, orient(item[ITEM_FIELDS.WIDTH:ITEM_FIELDS.HEIGHT+1], selected_orientation))
      item_orientation_vector[item[ITEM_FIELDS.UUID]] = selected_orientation

    self.orders.append(item_order_id)
    self.is_order_reversed.extend([False])
    return True

  def update(self, selected_EMS, rotated_item):

    # 1. place box in a EMS
    item_in_place = np.array(rotated_item)
    selected_min = np.array(selected_EMS[0])
    item_ems = [selected_min, selected_min + item_in_place]

    # 2. Generate new EMSs resulting from the intersection of the box
    for EMS in self.EMSs.copy():
      if overlapped(item_ems, EMS):

        # eliminate(EMS)
        for index, EMS_copy in enumerate(self.EMSs):
          if np.array_equal(EMS[0], EMS_copy[0]) and  np.array_equal(EMS[1], EMS_copy[1]):
            self.EMSs.pop(index)

        # six new EMSs in 3 dimensions
        x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
        x3, y3, z3 = item_ems[0]; x4, y4, z4 = item_ems[1]
        new_EMSs = [
            [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
            [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
            [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
        ]


        for new_EMS in new_EMSs:
          isValid = True

          # 3. Eliminate new EMSs which are totally inscribed by other EMSs
          for other_EMS in self.EMSs:
            if inscribed(new_EMS, other_EMS):
              isValid = False

          if isValid:
            self.EMSs.append(new_EMS)


  # Distance to the Front-Top-Right Corner
  def DFTRC_2(self, item):
      maxDist = -1
      selectedEMS = None
      selected_orientation = None

      for EMS in self.EMSs:
          D, W, H = self.dimensions
          for orientation in [1,2,3,4,5,6]:
              d, w, h = orient(item, orientation)
              if fitin((d, w, h), EMS):
                  x, y, z = EMS[0]
                  distance = pow(D-x-d, 2) + pow(W-y-w, 2) + pow(H-z-h, 2)

                  if distance > maxDist:
                      maxDist = distance
                      selectedEMS = EMS
                      selected_orientation = orientation
      return selectedEMS, selected_orientation


def orient(item, orientation=1):
  d, w, h = item
  if   orientation == 1: return (d, w, h)
  elif orientation == 2: return (d, h, w)
  elif orientation == 3: return (w, d, h)
  elif orientation == 4: return (w, h, d)
  elif orientation == 5: return (h, d, w)
  elif orientation == 6: return (h, w, d)


def overlapped(ems, EMS):
    if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
        return True
    return False

def inscribed(ems, EMS):
    if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
        return True
    return False


def fitin(item, EMS):
  # all dimension fit
  for d in range(3):
      if item[d] > EMS[1][d] - EMS[0][d]:
          return False
  return True
