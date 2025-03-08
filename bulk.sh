for cls in {0..1}
do
  for instance in {0..5}
  do
    echo "$cls $instance"
    python experiment_instance.py churn $cls $instance --confirm_measure S --confirm_threshold 0.15
    python experiment_instance.py churn $cls $instance --confirm_measure N --confirm_threshold 0.0
    python experiment_instance.py churn_telco $cls $instance --confirm_measure S --confirm_threshold 0.145
    python experiment_instance.py churn_telco $cls $instance --confirm_measure N --confirm_threshold 0.10
  done
done