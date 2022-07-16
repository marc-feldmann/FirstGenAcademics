
temp_ss1 = se_red_17_wod[se_red_17_wod["dem01_h"] == "männlich"]
temp_ss1 = temp_ss1["dem01_h"]
temp_ss2 = se_red_17_wod[se_red_17_wod["dem01_h"] == "weiblich"]
temp_ss2 = temp_ss2["dem01_h"]
# temp_ss2 = se_red_17_wod[se_red_17_wod["dem01_h"] == "divers"]
plt.bar(se_red_17_wod["dem99_c"], temp_ss1, color="r")
plt.bar(se_red_17_wod["dem99_c"], temp_ss2, bottom=temp_ss1, color="b")
plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
# fig.suptitle("Gender Distribution")
# temp_ss1 = se_red_17_wod[se_red_17_wod["dem99_c"] == 0]
# temp_ss2 = se_red_17_wod[se_red_17_wod["dem99_c"] == 1]
# # plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], ["0.00", "0.25", "0.50", "0.75", "1.00"])
# sns.histplot(ax=axes[0], x="dem01_h", multiple="dodge", stat="probability", discrete=True, data=temp_ss1)
# sns.histplot(ax=axes[1], x="dem01_h", multiple="dodge", stat="probability", discrete=True, data=temp_ss2)
# plt.show()

# sns.displot(se_red_17_wod, x="dem01_h", hue="dem99_c", stat="probability")
# plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00], ["0.00", "0.25", "0.50", "0.75", "1.00"])
# plt.show()


# visualize par07_h (Vater berufliche Position) - histogram
data_temp = se_red_17[se_red_17["par07_h"] != "nie berufstätig gewesen"]
p = sns.histplot(data=data_temp, x="par07_h", hue="dem99_c", hue_order=[0, 1], multiple="dodge", shrink=0.8, palette=["black", "grey"])
# the non-logarithmic labels you want
# p.set_yscale("log")
# p.set_yticks([1, 10, 100, 1000, 10000])
# p.set_yticklabels([1, 10, 100, 1000, 10000])
p.set_xticklabels(["civil servant", "clerk", "worker", "self-employed"])
p.set_xlabel("Father's Occupation")
p.set_ylabel("Counts")
plt.legend(title="", loc="upper right", labels=["first generation students", "other students"])
plt.show()