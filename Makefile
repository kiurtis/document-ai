VALET_NAME?="John Doe"
PLATE_NUMBER?=12-AB-34
FROM_CONCESSIONAIRE?=default_from_concessionaire
TO_CONCESSIONAIRE?=default_to_concessionaire
INPUT_FILE_PATH?=data/performances_data/valid_data/arval_classic_restitution_images/DH-427-VH_PV_RESTITUTION_DH-427-VH_p1.jpeg

# Command to run the Python script
run:
	@echo "Running document analysis..."
	@python main.py --valet_name $(VALET_NAME) --plate_number $(PLATE_NUMBER) --from_concessionaire $(FROM_CONCESSIONAIRE) --to_concessionaire $(TO_CONCESSIONAIRE) --input_file_path $(INPUT_FILE_PATH)
