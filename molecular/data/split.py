import random
import pyexcel as p
import xlrd2
import xlwt
import xlrd

total_num = 102
test_ratio = 0.3
test_num = int(total_num * test_ratio)

numbers = range(1, total_num)

random_numbers = random.sample(numbers, test_num)
random_numbers.sort()
n_random_numbers = []
for num in numbers:
    if num not in random_numbers:
        n_random_numbers.append(num)

# Open the Excel file
workbook = xlrd2.open_workbook('real_6_smiles_yields_product_pure_update.xlsx')
sheet = workbook.sheet_by_index(0)

# Read the rows
rows = []
for i in random_numbers:
    rows.append(sheet.row_values(i))

# Reverse the list
rows = rows[::-1]

# Create a new Excel file
workbook_out = xlwt.Workbook()
sheet_out = workbook_out.add_sheet('Sheet1')

# Write the rows to the new file
for i in range(len(rows)):
    for j in range(len(rows[i])):
        sheet_out.write(i, j, rows[i][j])

# Save the new Excel file
workbook_out.save('output.xls')

# Read the rows
rows = []
for i in n_random_numbers:
    rows.append(sheet.row_values(i))

# Reverse the list
rows = rows[::-1]

# Create a new Excel file
workbook_out = xlwt.Workbook()
sheet_out = workbook_out.add_sheet('Sheet1')

# Write the rows to the new file
for i in range(len(rows)):
    for j in range(len(rows[i])):
        sheet_out.write(i, j, rows[i][j])

# Save the new Excel file
workbook_out.save('output2.xls')

# 读取第一个Excel文件
book1 = xlrd2.open_workbook('real_6_smiles_yields_product_out.xlsx')
sheet1 = book1.sheet_by_index(0)

# 读取第二个Excel文件
book2 = xlrd.open_workbook('output2.xls')
sheet2 = book2.sheet_by_index(0)

# 创建一个新的Excel文件
book = xlwt.Workbook()
sheet = book.add_sheet('Sheet1')

# 将第一个Excel文件的内容复制到新的Excel文件中
for row in range(sheet1.nrows):
    for col in range(sheet1.ncols):
        sheet.write(row, col, sheet1.cell(row, col).value)

# 将第二个Excel文件的内容复制到新的Excel文件中
for row in range(sheet2.nrows):
    for col in range(sheet2.ncols):
        sheet.write(row + sheet1.nrows, col, sheet2.cell(row, col).value)

# 保存新的Excel文件
book.save('out1put.xls')

p.save_book_as(file_name='out1put.xls',
               dest_file_name='real_6_smiles_yields_product_2_train.xlsx')

# 读取第一个Excel文件
book1 = xlrd2.open_workbook('blank4.xlsx')
sheet1 = book1.sheet_by_index(0)

# 读取第二个Excel文件
book2 = xlrd.open_workbook('output.xls')
sheet2 = book2.sheet_by_index(0)

# 创建一个新的Excel文件
book = xlwt.Workbook()
sheet = book.add_sheet('Sheet1')

# 将第一个Excel文件的内容复制到新的Excel文件中
for row in range(sheet1.nrows):
    for col in range(sheet1.ncols):
        sheet.write(row, col, sheet1.cell(row, col).value)

# 将第二个Excel文件的内容复制到新的Excel文件中
for row in range(sheet2.nrows):
    for col in range(sheet2.ncols):
        sheet.write(row + sheet1.nrows, col, sheet2.cell(row, col).value)

# 保存新的Excel文件
book.save('out1put.xls')

p.save_book_as(file_name='out1put.xls',
               dest_file_name='real_6_smiles_yields_product_2_test.xlsx')

# 读取第一个Excel文件
book1 = xlrd2.open_workbook('blank4.xlsx')
sheet1 = book1.sheet_by_index(0)

# 读取第二个Excel文件
book2 = xlrd.open_workbook('output2.xls')
sheet2 = book2.sheet_by_index(0)

# 创建一个新的Excel文件
book = xlwt.Workbook()
sheet = book.add_sheet('Sheet1')

# 将第一个Excel文件的内容复制到新的Excel文件中
for row in range(sheet1.nrows):
    for col in range(sheet1.ncols):
        sheet.write(row, col, sheet1.cell(row, col).value)

# 将第二个Excel文件的内容复制到新的Excel文件中
for row in range(sheet2.nrows):
    for col in range(sheet2.ncols):
        sheet.write(row + sheet1.nrows, col, sheet2.cell(row, col).value)

# 保存新的Excel文件
book.save('out1put.xls')

p.save_book_as(file_name='out1put.xls',
               dest_file_name='real_6_smiles_yields_product_2_train_2.xlsx')