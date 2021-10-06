import logging

from faker import Faker
import pandas as pd

LOGGER = logging.getLogger(__name__)

class FakerFactory:

  @classmethod
  def apply(cls, field_anonymize: dict, num_rows: int, df_gen): #The df CAN be a single df or a dict {name: df}
    fake = Faker()

    data = []
    columns = []

    faker_plus = {} # {OG_VALUE: replacement} # all in one dict damit es überall passend replaced wird
    faker_plus_attr = {}

    for key, value in field_anonymize.items():
      if isinstance(value, dict):
        for field, col in value.items():
          # Hier fehlt noch die unterscheidung ob Faker+ aber wtf is hier col?
          print("Debugger log to find out what col is:")
          print(col)
          print(field)
          columns.append(col)
      else:
        if '+' not in value:
          columns.append(key)
        else:
          faker_plus_attr[key] = value

    for row in range(0, num_rows):
      row_data = []

      for key, faker_type in field_anonymize.items():
        if isinstance(faker_type, dict):
          for field, faker_type in faker_type.items():
            if faker_type == 'name':
              row_data.append(fake.name_nonbinary())
            elif faker_type == 'job':
              row_data.append(fake.job())
            elif faker_type == 'postcode':
              row_data.append(fake.postcode())
        else:
          if faker_type == 'name':
            row_data.append(fake.name_nonbinary())
          elif faker_type == 'job':
            row_data.append(fake.job())
          elif faker_type == 'postcode':
            row_data.append(fake.postcode())          

      data.append(row_data)
    

    for col, faker_type in faker_plus_attr.items():
      if isinstance(df_gen, dict):
        print("multitable")
      else:
        for unique in df_gen[col].unique():
          if not unique in faker_plus:
            if 'name' in faker_type:
              faker_plus[unique] = fake.name()

    print(faker_plus)
    return (pd.DataFrame(data, columns=columns), faker_plus, faker_plus_attr.keys())

  @classmethod
  def get_suggestions(cls, df: pd.DataFrame):
    return None