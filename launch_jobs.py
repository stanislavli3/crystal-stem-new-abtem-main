
import subprocess
import os
import csv


MATERIAL_CSV_PATH = r'csv/Cu-C.csv'  # CSV File downloaded from materials project.org

LOCAL_RUN_FILE = "main.py"          # main simulation code 
LOCAL_FOLDER_PATH = "/home/bellefonte/Desktop/simulation/crystal-stem-new-abtem-main/"
LOCAL_ENV_PATH = "/home/bellefonte/Desktop/simulation/crystal-stem-new-abtem-main/env/"




def parse_csv():
    """Get material id from csv file 
    Args:
        MATERIAL_CSV_PATH (String): Path to csv file 
    Returns:
        material_id_list: dict{key: mat_id, value: submit_status}
    """
    material_id_list = []
    try: 
        with open(MATERIAL_CSV_PATH) as material_csv:
            csv_reader = csv.reader(material_csv)
            header = next(csv_reader)
            material_id_index = header.index("Material ID")
            material_id_list = [(row[material_id_index],0) for row in csv_reader]
            
            return dict(material_id_list)
    except Exception as e:
        print(f"File reading error: {e}")

def main():
    
    # parse csv file 
    material_id_list = parse_csv()
       
    # get list of unprocessed material ids 
    # todo_id_list = [entry for entry in material_id_list if entry[1] == 0]
    todo_id_list = [key for key, value in material_id_list.items() if value == 0]
    if not todo_id_list:
        print("Finish submitting. No jobs left to submit!!!")
        return
    
    total_materials = len(todo_id_list)
    
    for index in range(0, total_materials):
        material_id = todo_id_list[index]
        
        # if main file is not running, launch another job 
        if subprocess.run(f"pgrep -f {LOCAL_RUN_FILE}", shell=True, capture_output=True) != 0:

            # activate environment 
            activate_command = f"conda activate {LOCAL_ENV_PATH}"
            subprocess.run(activate_command, shell=True)
            os.chdir(LOCAL_FOLDER_PATH)
            
            subprocess.run(['python3', LOCAL_RUN_FILE, material_id])
            print(f"Finished running job for Material ID: {material_id}")
            
            
            # change status in list 
            material_id_list[material_id] = 1
                   
            # except subprocess.CalledProcessError as e:
            #     print(f"Error running local job for Material ID {material_id}: {e}")
                

if __name__ == "__main__":
    main()
