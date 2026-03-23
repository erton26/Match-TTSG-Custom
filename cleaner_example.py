from match_ttsg.text.cleaners import japanese_cleaners, japanese_accent_cleaners

text = "スーパーで買う野菜よりも味もいいような気がして"

result_jc = japanese_cleaners(text)
result_jac = japanese_accent_cleaners(text)

print(result_jc)
print(result_jac)