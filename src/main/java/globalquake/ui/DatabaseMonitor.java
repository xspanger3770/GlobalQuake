package globalquake.ui;

import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.ConcurrentModificationException;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTree;
import javax.swing.border.EmptyBorder;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;

import com.morce.globalquake.database.Channel;
import com.morce.globalquake.database.Network;
import com.morce.globalquake.database.Station;

import globalquake.database.StationManager;
import globalquake.main.Main;

public class DatabaseMonitor extends JFrame {

	private static final long serialVersionUID = 1L;
	private JPanel contentPane;
	private StationManager stationManager;
	private JLabel lblVersion;
	private JLabel lblLastUpdate;
	private JProgressBar progressBar2;
	private JProgressBar progressBar1;
	private JButton btnLaunch;
	private JButton btnSelectStations;
	private JLabel lblNetworks;
	private JLabel lblStations;
	private JLabel lblChannels;
	private JLabel lblSelected;
	private JLabel lblAvailable;
	private DefaultTreeModel defaultTreeModel;
	private JButton btnUpdate;
	private JTree tree;
	private StationSelect stationSelect;
	private Thread uiThread;

	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

	public DatabaseMonitor(StationManager stationManager, Main globalQuake) {
		this.stationManager = stationManager;
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		contentPane = new JPanel();
		contentPane.setPreferredSize(new Dimension(460, 330));
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(null);

		btnLaunch = new JButton("Launch " + Main.fullName);
		btnLaunch.setEnabled(false);
		btnLaunch.setBounds(230, 285, 218, 32);
		btnLaunch.setMargin(new Insets(0, 0, 0, 0));
		contentPane.add(btnLaunch);

		btnLaunch.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				if (stationManager.getSelectedStations() == 0) {
					String[] options = { "Cancel", "Yes" };
					int n = JOptionPane.showOptionDialog(DatabaseMonitor.this, "Launch with no selected stations?",
							"Confirm", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, options,
							options[0]);
					if (n == 0) {
						return;
					}
				}
				uiThread.interrupt();
				DatabaseMonitor.this.dispose();
				globalQuake.launch();
			}
		});

		lblVersion = new JLabel("Database Version: Loading...");
		lblVersion.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblVersion.setBounds(10, 25, 238, 20);
		contentPane.add(lblVersion);

		lblLastUpdate = new JLabel("Last Update: Loading...");
		lblLastUpdate.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblLastUpdate.setBounds(10, 5, 238, 20);
		contentPane.add(lblLastUpdate);

		progressBar1 = new JProgressBar();
		progressBar1.setIndeterminate(true);
		progressBar1.setString("0% - Waiting...");
		progressBar1.setStringPainted(true);
		progressBar1.setValue(0);
		progressBar1.setBounds(10, 200, 438, 32);
		contentPane.add(progressBar1);

		progressBar2 = new JProgressBar();
		progressBar2.setIndeterminate(true);
		progressBar2.setString("0% - Waiting...");
		progressBar2.setStringPainted(true);
		progressBar2.setBounds(10, 240, 438, 32);
		contentPane.add(progressBar2);

		btnSelectStations = new JButton("Select Stations");
		btnSelectStations.setEnabled(false);
		btnSelectStations.setBounds(10, 285, 210, 32);
		contentPane.add(btnSelectStations);

		btnSelectStations.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				EventQueue.invokeLater(new Runnable() {

					@Override
					public void run() {
						if (stationManager == null) {
							return;
						}
						DatabaseMonitor.this.setEnabled(false);
						stationSelect = new StationSelect(stationManager);
						stationSelect.setVisible(true);
						stationSelect.addWindowListener(new WindowAdapter() {
							@Override
							public void windowClosing(WindowEvent e) {
								DatabaseMonitor.this.setEnabled(true);
							}
						});
					}
				});
			}
		});

		lblNetworks = new JLabel("Networks: Loading...");
		lblNetworks.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblNetworks.setBounds(10, 45, 238, 20);
		contentPane.add(lblNetworks);

		lblStations = new JLabel("Stations: Loading...");
		lblStations.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblStations.setBounds(10, 65, 210, 20);
		contentPane.add(lblStations);

		lblChannels = new JLabel("Channels: Loading...");
		lblChannels.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblChannels.setBounds(10, 85, 210, 20);
		contentPane.add(lblChannels);

		lblSelected = new JLabel("Selected stations: Loading...");
		lblSelected.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblSelected.setBounds(10, 125, 210, 20);
		contentPane.add(lblSelected);

		lblAvailable = new JLabel("Available stations: Loading...");
		lblAvailable.setFont(new Font("Tahoma", Font.BOLD, 12));
		lblAvailable.setBounds(10, 105, 210, 20);
		contentPane.add(lblAvailable);

		DefaultMutableTreeNode wait = new DefaultMutableTreeNode("Loading...");

		defaultTreeModel = new DefaultTreeModel(wait);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(258, 5, 190, 183);
		contentPane.add(scrollPane);
		tree = new JTree(defaultTreeModel);
		tree.setEnabled(false);
		scrollPane.setViewportView(tree);

		btnUpdate = new JButton("Update");
		btnUpdate.setBounds(10, 155, 238, 32);
		btnUpdate.setEnabled(false);
		contentPane.add(btnUpdate);

		btnUpdate.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent e) {
				if (stationManager != null && stationManager.getDatabase() != null) {
					new Thread() {
						public void run() {
							stationManager.update(true);
						};
					}.start();
				}
			}
		});

		pack();
		setTitle("Station Database");
		setResizable(false);
		setLocationRelativeTo(null);

		uiThread = new Thread("Database Monitor UI Updater") {
			public void run() {
				while (true) {
					try {
						sleep(50);
					} catch (InterruptedException e) {
						break;
					}
					EventQueue.invokeLater(new Runnable() {

						public void run() {
							updateFrame();
						}

					});
				}
			};
		};
		uiThread.start();

		this.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				uiThread.interrupt();
			}
		});
	}

	public void confirmDialog(String title, String message, int optionType, int messageType, String... options) {
		int n = JOptionPane.showOptionDialog(this, message, title, optionType, messageType, null, options, options[0]);
		if (n == 0 && options.length > 1) {
			System.exit(0);
		}
	}

	private void updateFrame() {
		int state = stationManager.getState();
		if (stationManager == null) {
			lblVersion.setText("FATAL ERROR");
		} else {
			if (stationManager.getDatabase() == null) {
				return;
			}
			if (state < StationManager.DONE && btnSelectStations.isEnabled()) {
				btnSelectStations.setEnabled(false);
				btnLaunch.setEnabled(false);
				btnUpdate.setEnabled(false);
				tree.setEnabled(false);
			}
			if (state <= 0) {
				lblVersion.setText("Database version: Loading...");
				lblNetworks.setText("Networks: Loading...");
				lblStations.setText("Stations: Loading...");
				lblChannels.setText("Channels: Loading...");
				lblSelected.setText("Selected: Loading...");
				lblLastUpdate.setText("Last update: Loading...");
				lblAvailable.setText("Available: Loading...");
			}

			if (state >= StationManager.CHECKING_AVAILABILITY) {
				if (progressBar2.isIndeterminate()) {
					progressBar2.setIndeterminate(false);
				}
			} else if (!progressBar2.isIndeterminate()) {
				progressBar2.setIndeterminate(true);
			}

			if (state > 0) {
				if (progressBar1.isIndeterminate()) {
					progressBar1.setIndeterminate(false);
				}
			} else {
				if (!progressBar1.isIndeterminate()) {
					progressBar1.setIndeterminate(true);
				}
			}

			if (state > 0) {
				if (stationManager.getDatabase().getDatabaseVersion() == StationManager.DATABASE_VERSION) {
					lblVersion.setText("<html>Database version: <font color='green'>"
							+ stationManager.getDatabase().getDatabaseVersion() + "</font></html>");
				} else {
					lblVersion.setText("<html>Database version: <font color='red'>"
							+ stationManager.getDatabase().getDatabaseVersion() + " -> "
							+ StationManager.DATABASE_VERSION + "</font></html>");
				}
				Calendar c = Calendar.getInstance();
				c.setTimeInMillis(stationManager.getDatabase().getLastUpdate());
				lblLastUpdate.setText("<html>Last update: "
						+ (state == StationManager.UPDATING_DATABASE ? "<font color='green'>Updating now...</font>"
								: "<font color='" + colAge(c) + "'>" + dateFormat.format(c.getTime()) + "</font>")
						+ "</html>");
				progressBar1.setValue((int) (stationManager.updating_progress * 100));
				progressBar1.setString(
						(int) (stationManager.updating_progress * 100) + "% - " + stationManager.updating_string);

				progressBar2.setValue((int) (stationManager.availability_progress * 100));
				progressBar2.setString((int) (stationManager.availability_progress * 100) + "% - "
						+ stationManager.availability_string);

				int nets = 0;
				int stats = 0;
				int chans = 0;
				int ava = 0;
				int sel = 0;

				try {
					for (Network n : stationManager.getDatabase().getNetworks()) {
						nets++;
						for (Station s : n.getStations()) {
							stats++;
							boolean av = false;
							boolean se = false;
							for (Channel ch : s.getChannels()) {
								chans++;
								if (ch.isAvailable() && !av) {
									ava++;
									av = true;
								}
								if (ch.isSelected() && !se) {
									sel++;
									se = true;
								}
							}
						}
					}
				} catch (ConcurrentModificationException ex) {
					// sometimes happen during update
					return;
				}

				lblNetworks.setText("Networks: " + nets);
				lblStations.setText("Stations: " + stats);
				lblChannels.setText("Channels: " + chans);
				lblAvailable.setText("Available stations: " + ava);
				if (state >= StationManager.DONE) {
					lblSelected.setText("Selected stations: " + sel);
				}
				if (state >= StationManager.DONE && !btnSelectStations.isEnabled()) {
					btnSelectStations.setEnabled(true);
					btnLaunch.setEnabled(true);
					btnUpdate.setEnabled(true);
					tree.setEnabled(true);
					defaultTreeModel.setRoot(createRoot());
				}
			}
		}
		// revalidate();
		repaint();
	}

	private String colAge(Calendar c) {
		double ageDays = (System.currentTimeMillis() - c.getTimeInMillis()) / (1000.0 * 60 * 60 * 24);
		if (ageDays >= 7) {
			return "red";
		}
		if (ageDays >= 3) {
			return "orange";
		}
		return "green";
	}

	private DefaultMutableTreeNode createRoot() {
		DefaultMutableTreeNode networksNode = new DefaultMutableTreeNode("Networks");
		for (Network n : stationManager.getDatabase().getNetworks()) {
			DefaultMutableTreeNode netwNode = new DefaultMutableTreeNode(n.getNetworkCode());
			for (Station s : n.getStations()) {
				DefaultMutableTreeNode statNode = new DefaultMutableTreeNode(s.getStationCode());
				for (Channel ch : s.getChannels()) {
					DefaultMutableTreeNode chanNode = new DefaultMutableTreeNode(
							ch.getName() + " " + ch.getLocationCode());
					statNode.add(chanNode);
				}
				netwNode.add(statNode);
			}

			networksNode.add(netwNode);
		}
		return networksNode;
	}

	public StationManager getStationManager() {
		return stationManager;
	}
}
