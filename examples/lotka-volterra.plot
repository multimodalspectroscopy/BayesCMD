set terminal x11 persist
set size .75,.75
plot "lv.detail" using 2:3 with lines ti "Prey", "lv.dat" using 2:4 with lines ti "Predator"
