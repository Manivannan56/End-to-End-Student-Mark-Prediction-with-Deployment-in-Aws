from setuptools import find_packages,setup
from typing import List

Hyphen_e_dot="-e."

def get_requirements(File_Path:str)->List[str]:
   "This function will return the list of requirements" 
   with open(File_Path) as file_obj:
      requirements=file_obj.readlines()
      requirements=[req.replace("\n"," ") for req in requirements]

      if Hyphen_e_dot in requirements:
         requirements.remove(Hyphen_e_dot)

   return requirements



setup(
name='ML_Project',
version='0.0.1',
author='Manivannan',
author_email='manivannan19056@ece.ssn.edu.in',
packages=find_packages(),
install_requires=get_requirements('/Users/manivannans/Desktop/End_to_End_Machine_learning_project/requirements.txt')


)