import xlsxwriter

class FuzzyType:
    def __init__(self, type, use_sigma_scale, use_height_scale):
        self.type = type
        self.use_sigma_scale = use_sigma_scale
        self.use_height_scale = use_height_scale

n_inputs = [5, 15, 30]
n_memberships = [3, 7, 15]
fuzzy_types = [FuzzyType(1, 0, 1), FuzzyType(2,0,1), FuzzyType(2,1,1), FuzzyType(2,1,0)]
datasets = {'MNIST':1, 'FashionMNIST':2, 'QuickDraw':4}

def create_sheet(workbook, name, data, alpha):
    worksheet = workbook.add_worksheet()
    worksheet.name = name
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',})

    # Merge Necesssary cells
    worksheet.merge_range('C1:D1', '', merge_format)
    worksheet.merge_range('E1:F1', '', merge_format)
    worksheet.merge_range('G1:H1', '', merge_format)
    worksheet.merge_range('I1:J1', '', merge_format)
    # Set headers
    worksheet.set_column('A:Z', 10)
    worksheet.set_default_row(20)
    worksheet.write('C1', "T1", merge_format)
    worksheet.write('E1', "T2_A", merge_format)
    worksheet.write('G1', "T2_B", merge_format)
    worksheet.write('I1', "T2_C", merge_format)

    worksheet.write('C2', "max", merge_format)
    worksheet.write('D2', "avg", merge_format)
    worksheet.write('E2', "max", merge_format)
    worksheet.write('F2', "avg", merge_format)
    worksheet.write('G2', "max", merge_format)
    worksheet.write('H2', "avg", merge_format)
    worksheet.write('I2', "max", merge_format)
    worksheet.write('J2', "avg", merge_format)

    worksheet.merge_range('A3:A5', '5', merge_format)
    worksheet.merge_range('A6:A8', '15', merge_format)
    worksheet.merge_range('A9:A11', '30', merge_format)

    format = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter', })

    worksheet.write('B3', '3',format)
    worksheet.write('B4', '7', format)
    worksheet.write('B5', '15', format)
    worksheet.write('B6', '3', format)
    worksheet.write('B7', '7', format)
    worksheet.write('B8', '15', format)
    worksheet.write('B9', '3', format)
    worksheet.write('B10', '7', format)
    worksheet.write('B11', '15', format)

    for n_input in n_inputs:
        for n_membership in n_memberships:
            for f_type in fuzzy_types:
                exp_id = "{}_{}_{}_{}_{}_{}_{}_{}".format(datasets[name], f_type.type, n_input, n_membership,
                                                          alpha, 2.5, f_type.use_sigma_scale,
                                                          f_type.use_height_scale)
                if exp_id in data.keys():
                    max_acc = data[exp_id]["max_acc"]
                    avg_acc = data[exp_id]["avg_acc"]

                    n_input_index = n_inputs.index(n_input)
                    n_membership_index = n_memberships.index(n_membership)
                    fuzzy_index = fuzzy_types.index(f_type)
                    row_index = 2 + n_input_index * 3 + n_membership_index
                    col_index = 2 + fuzzy_index * 2
                    max_index = col_index
                    avg_index = col_index + 1
                    worksheet.write(row_index, max_index, max_acc)
                    worksheet.write(row_index, avg_index, avg_acc)




if __name__ == "__main__":
    import json
    # For Know Dist
    workbook = xlsxwriter.Workbook("table.xlsx")
    f = open("./results.json")
    data = json.load(f)
    f.close()

    create_sheet(workbook, "MNIST", data, 0.75)
    create_sheet(workbook, "FashionMNIST", data, 0.75)
    create_sheet(workbook, "QuickDraw", data, 0.75)
    workbook.close()

    # Without Know Dist
    workbook = xlsxwriter.Workbook("table_no_know.xlsx")

    create_sheet(workbook, "MNIST", data, 0.0)
    create_sheet(workbook, "FashionMNIST", data, 0.0)
    create_sheet(workbook, "QuickDraw", data, 0.0)
    workbook.close()

