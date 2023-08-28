package globalquake.ui.settings;

import java.awt.*;
import java.util.LinkedList;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

import org.tinylog.Logger;

public class SettingsFrame extends JFrame{

	private final List<SettingsPanel> panels = new LinkedList<>();
	private JTabbedPane tabbedPane;

	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
            try {
                new SettingsFrame(null).setVisible(true);
            } catch (Exception e) {
				Logger.error(e);
            }
        });
	}

	public SettingsFrame(Component parent) {
		initialize(parent);
	}

	private void initialize(Component parent) {
		setTitle("GlobalQuake Settings");
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setMinimumSize(new Dimension(500, 400));
		JPanel panel = new JPanel(new BorderLayout());
		setContentPane(panel);

		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		panel.add(tabbedPane, BorderLayout.CENTER);

		JPanel panel_1 = new JPanel();
		panel.add(panel_1, BorderLayout.SOUTH);

		JButton btnCancel = new JButton("Cancel");
		panel_1.add(btnCancel);

		btnCancel.addActionListener(e -> dispose());

		JButton btnSave = new JButton("Save");
		panel_1.add(btnSave);

		btnSave.addActionListener(e -> {
            for (SettingsPanel panel1 : panels) {
                try {
                    panel1.save();
                } catch (Exception ex) {
                    error(ex);
                    return;
                }
            }
            Settings.save();
			dispose();
        });

		addPanels();

		pack();
		setLocationRelativeTo(parent);
	}

	protected void error(Exception e) {
		JOptionPane.showMessageDialog(this, e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
	}

	private void addPanels() {
		panels.add(new GeneralSettingsPanel());
		panels.add(new PerformanceSettingsPanel());
		panels.add(new AppearanceSettingsPanel());
		panels.add(new HypocenterAnalysisSettingsPanel());

		for (SettingsPanel panel : panels) {
			tabbedPane.addTab(panel.getTitle(), panel);
		}
	}

}
