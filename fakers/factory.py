import logging

from faker import Faker
import pandas as pd

LOGGER = logging.getLogger(__name__)

class FakerFactory:

  @classmethod
  def apply(cls, field_anonymize: dict, num_rows: int):
    fake = Faker()

    data = []
    columns = []

    for key, value in field_anonymize.items():
      if isinstance(value, dict):
        for field, col in value.items():
          columns.append(col)
      else:
        columns.append(key)

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
    
    print(columns)
    return pd.DataFrame(data, columns=columns)

  @classmethod
  def get_suggestions(cls, df: pd.DataFrame):
    return None