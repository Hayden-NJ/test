def oswalk_file_paths(self,top_path,extension=['.pdf']):
	file_paths = []
	for i in os.walk(top_path):
		if i[2]:
			for j in i[2]:
				if os.path.splitext(j)[-1] in extension:
					concatenate_path = os.path.join(i[0],j)
					file_path_list.append(concatenate_path)
	return file_paths