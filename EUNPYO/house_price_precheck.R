'''
고칠 점들
1. 이상치 없애기

'''
df=read.csv("//Users//pio//work//house_price.csv",header=T)
class(df)

colnames(df)

attach(df)
CONDITION <- factor(CONDITION,levels=c('Poor','Fair','Average','Good','Very Good','Excellent'),ordered=T)
CONDITION

# 우선 간단하게 REMODEL_YEAR 수정(2019년 기준으로 리모델링한 후 지난 시간)
REMODEL_YEAR <- 2019-REMODEL_YEAR
plot(PRICE~REMODEL_YEAR)

# story 값도 반올림하는 편이 나을 듯
STORIES <- round(STORIES)

# 상관관계 보기. 9번째 변수는 CONDITION이라 제외
cor(df[-9])


# 이산형 변수의 경우 각각 변수의 숫자별로 가격에 유의미한 차이 있는 지 확인
boxplot(PRICE~BATHROOMS) # pick
boxplot(PRICE~BATHROOMS,outline=F) 
table(BATHROOMS)

boxplot(PRICE~NUM_UNITS)
boxplot(PRICE~NUM_UNITS,outline=F) # pick
table(NUM_UNITS)

boxplot(PRICE~ROOMS)
boxplot(PRICE~ROOMS,outline=F)
table(ROOMS)

boxplot(PRICE~BEDROOMS)
boxplot(PRICE~BEDROOMS,outline=F)
table(BEDROOMS)

boxplot(PRICE~STORIES)
boxplot(PRICE~STORIES,outline=F) # pick
table(STORIES)

boxplot(PRICE~CONDITION)
boxplot(PRICE~CONDITION,outline=F) #pick
table(CONDITION)

boxplot(PRICE~KITCHENS)
boxplot(PRICE~KITCHENS,outline=F) #pick
table(KITCHENS)

boxplot(PRICE~FIREPLACES)
boxplot(PRICE~FIREPLACES,outline=F) #pick
table(FIREPLACES)

boxplot(PRICE~STORIES)
boxplot(PRICE~STORIES,outline=F) #pick


# 연속형 자료 : 원자료와  log 변환한 자료 비교
library(nortest)
par(mfrow=c(1,2))

hist(PRICE,breaks=150)
hist(log(PRICE),breaks=150)
ad.test(PRICE)
ad.test(log(PRICE))

barplot(REMODEL_YEAR)

hist(BUILDING_AREA,breaks=150)
hist(log(BUILDING_AREA),breaks=150)
ad.test(BUILDING_AREA)
ad.test(log(BUILDING_AREA))

hist(LAND_AREA,breaks=150)
hist(log(LAND_AREA),breaks=150)
ad.test(LAND_AREA)
ad.test(log(LAND_AREA))





