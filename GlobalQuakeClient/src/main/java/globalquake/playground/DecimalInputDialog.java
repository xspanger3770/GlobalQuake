package globalquake.playground;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;

public class DecimalInputDialog extends JDialog {
    private final List<DecimalInput> decimalInputs;
    private JSlider[] sliders;

    public DecimalInputDialog(JFrame parent, String title, java.util.List<DecimalInput> decimalInputs, Runnable action) {
        super(parent, title, true);
        this. decimalInputs = decimalInputs;

        sliders = new JSlider[decimalInputs.size()];

        setLayout(new GridLayout(decimalInputs.size() + 1, 1, 10, 10));

        for (int i = 0; i < decimalInputs.size(); i++) {
            var decimalInput = decimalInputs.get(i);
            double pct = (decimalInput.getValue() - decimalInput.getMin()) / (decimalInput.getMax() - decimalInput.getMin());
            sliders[i] = new JSlider(SwingConstants.HORIZONTAL, 0, 10000, (int) Math.round(10000 * pct));
            sliders[i].setPaintTicks(true);
            sliders[i].setMajorTickSpacing(1000);
            sliders[i].setMinorTickSpacing(200);
            sliders[i].setPreferredSize(new Dimension(500,40));
            // todo

            JPanel panel = new JPanel(new BorderLayout());
            JLabel field = new JLabel("%s: %.1f".formatted(decimalInput.getName(), getValue(i)));
            panel.add(field, BorderLayout.NORTH);
            panel.add(sliders[i], BorderLayout.CENTER);

            int finalI = i;
            sliders[i].addChangeListener(new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent changeEvent) {
                    decimalInputs.get(finalI).setValue(getValue(finalI));
                    field.setText("%s: %.1f".formatted(decimalInputs.get(finalI).getName(), getValue(finalI)));
                }
            });

            add(panel);
        }

        JButton okButton = new JButton("OK");
        JButton cancelButton = new JButton("Cancel");

        okButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
                action.run();
            }
        });

        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                dispose();
            }
        });

        JPanel buttonsPanel = new JPanel();

        buttonsPanel.add(cancelButton);
        buttonsPanel.add(okButton);

        add(buttonsPanel);

        pack();
        setLocationRelativeTo(parent);
        setVisible(true);
    }

    private double getValue(int i) {
        DecimalInput input = decimalInputs.get(i);
        double pct = sliders[i] . getValue()  / 10000.0;
        return input.getMin() + (input.getMax() - input.getMin()) * pct;
    }


    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                java.util.List<DecimalInput> inputs = new ArrayList<>();
                inputs.add(new DecimalInput("Magnitude", 0, 10, 4.0));
                inputs.add(new DecimalInput("Depth", 0, 750, 10.0));
                JFrame frame = new JFrame();
                DecimalInputDialog dialog = new DecimalInputDialog(frame, "Choose parameters", inputs, () -> System.err.println("ok"));
                dialog.setVisible(true);

                dialog.addWindowListener(new WindowAdapter() {
                    @Override
                    public void windowClosing(WindowEvent e) {
                        System.exit(0);
                    }
                });
            }
        });
    }
}
