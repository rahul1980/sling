BEGIN { num = 0; }
/^AvgDelegateLoss/ {
  c=$3;
  n=$5;
  l=$NF;
  count[c]++;
  file = "/tmp/plot." c ".data"
  if (count[c] == 1) system("rm -f " file);
  print n " " l >> file;
  if (num < c) num = c;
}
/^BatchLoss/ {
  n=$6;
  l=$8;
  count["total"]++;
  file = "/tmp/plot.total.data"
  if (count["total"] == 1) system("rm -f " file);
  print n " " l >> file;
}
/^Eval metric/ {
  n=$4;
  l=$6 ;
  count["eval"]++;
  file = "/tmp/plot.eval.data"
  if (count["eval"] == 1) system("rm -f " file);
  print n " " l >> file;
}
END {
  file = "/tmp/gnuplot.script"
  system("rm -f " file);
  print "Generated /tmp/plot.total.data"
  print "Generated /tmp/plot.eval.data"
  print "set terminal png" >> file
  print "set output '/tmp/plot.png'" >> file
  print "set y2tics" >> file
  print "set y2label \"F1\"" >> file
  print "set ytics nomirror" >> file
  print "set y2tics nomirror" >> file
  print "set ylabel \"Loss\"" >> file
  print "set xlabel \"#Train Sent. Seen\"" >> file
  print "plot \"/tmp/plot.total.data\" using 1:2 title 'Total' lw 2 w lp smooth csplines,\\" >> file
  print "     \"/tmp/plot.eval.data\" using 1:2 title 'Slot F1' axis x1y2 lw 2 w l smooth csplines,\\" >> file
  for (i=0; i <= num; i++) {
    s = ",\\"
    if (i == num) s = ""
  print "Generated /tmp/plot." i ".data"
    print "     \"/tmp/plot." i ".data\" using 1:2 title 'Delegate" i "' lw 2 w l smooth csplines" s >> file
  }
  print "set terminal x11" >> file
  print "set output" >> file
  print "replot" >> file
  print "pause -1" >> file
  print "Wrote gnuplot commands to " file
  print "Run it via 'gnuplot -c " file "'"
  print "It will generate a plot in /tmp/plot.png"
}

