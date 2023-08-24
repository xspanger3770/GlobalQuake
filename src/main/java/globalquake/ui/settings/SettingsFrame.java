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

public class SettingsFrame {

	private JFrame frame;

	private final List<SettingsPanel> panels = new LinkedList<>();
	private JTabbedPane tabbedPane;

	public static void main(String[] args) {
		EventQueue.invokeLater(() -> {
            try {
                SettingsFrame window = new SettingsFrame(null);
                window.frame.setVisible(true);
            } catch (Exception e) {
				Logger.error(e);
            }
        });
	}

	public SettingsFrame(Component parent) {
		initialize(parent);
	}

	private void initialize(Component parent) {
		frame = new JFrame("GlobalQuake Settings");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setMinimumSize(new Dimension(400, 300));
		JPanel panel = new JPanel(new BorderLayout());
		frame.setContentPane(panel);

		tabbedPane = new JTabbedPane(JTabbedPane.TOP);
		panel.add(tabbedPane, BorderLayout.CENTER);

		JPanel panel_1 = new JPanel();
		panel.add(panel_1, BorderLayout.SOUTH);

		JButton btnCancel = new JButton("Cancel");
		panel_1.add(btnCancel);

		btnCancel.addActionListener(e -> frame.dispose());

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
        });

		addPanels();

		frame.pack();
		frame.setLocationRelativeTo(parent);
	}

	protected void error(Exception e) {
		JOptionPane.showMessageDialog(frame, e.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
	}

	private void addPanels() {
		panels.add(new GeneralSettingsPanel());

		for (SettingsPanel panel : panels) {
			tabbedPane.addTab(panel.getTitle(), panel);
		}
	}

	public static void show() {
		main(null);
	}

}
