import React, { useEffect, useState } from 'react'

import { useParams } from 'react-router-dom'

import {
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
  CCard,
  CCardHeader,
  CCardBody,
} from '@coreui/react'
import axios from 'axios'

import { API_URL } from 'src/const'

const defaultUser = {
  userId: '',
  fullName: '',
  email: '',
}

const User = () => {
  const [user, setUser] = useState(defaultUser)
  const [cases, setCases] = useState([])
  const [apiKeys, setApiKeys] = useState([])
  const params = useParams()

  const handleCreateCaseButtonClick = async () => {
    try {
      const data = {
        apiKey: '62ab01130c0a0b810fe16552:PFxJs5nZUxo1BRlFHMsidA',
        type: 'DAMAGE_ASSESSMENT',
        userId: user.id,
      }
      const res = await axios.post(`${API_URL}/case`, data)
      if (res.status === 200) {
        window.location.reload()
      } else {
        console.log(res)
      }
    } catch (e) {
      console.log(e)
    }
  }

  const handleApiKeyButtonClick = async () => {
    try {
      const data = {
        userId: user.id,
        expirationDays: '7',
      }
      const res = await axios.post(`${API_URL}/api-key`, data)
      if (res.status === 200) {
        window.location.reload()
      } else {
        console.log(res)
      }
    } catch (e) {
      console.log(e)
    }
  }

  const fetchUser = async () => {
    const url = `${API_URL}/user/${params.userId}`
    try {
      const res = await axios.get(url)
      const resData = res.data
      if (res.status === 200) {
        const user = resData.data
        setUser(user)
      } else {
        console.log(res)
      }
    } catch (e) {
      console.log(e)
    }
  }

  const fetchCases = async () => {
    try {
      const url = `${API_URL}/case?userId=${user.id}`
      const caseRes = await axios.get(url)
      const caseResJson = caseRes.data
      const cases = caseResJson.data
      setCases(cases)
    } catch (e) {
      console.log(e)
      setCases([])
    }
  }
  const fetchApiKeys = async () => {
    try {
      const url = `${API_URL}/api-key?userId=${user.id}`
      const apiKeyRes = await axios.get(url)
      const apiKeyResJson = apiKeyRes.data
      const apiKeys = apiKeyResJson.data
      setApiKeys(apiKeys)
    } catch (e) {
      console.log(e)
      setApiKeys([])
    }
  }

  const handleDeleteCaseClick = async (caseId) => {
    const url = `${API_URL}/case/${caseId}`
    try {
      const res = await axios.delete(url)
      if (res.status === 200) {
        window.location.reload()
      }
    } catch (e) {
      console.log(e)
    }
  }

  const handleApiKeyDeleteClick = async (apiKeyId) => {
    const url = `${API_URL}/api-key/${apiKeyId}`
    try {
      const res = await axios.delete(url)
      if (res.status === 200) {
        window.location.reload()
      }
    } catch (e) {
      console.log(e)
    }
  }

  useEffect(() => {
    fetchUser()
  }, [])

  useEffect(() => {
    fetchCases()
    fetchApiKeys()
  }, [user])

  return (
    <>
      <div className="container">
        <div className="row">
          {/* USER DETAILS COL */}
          <div className="col col-md-4">
            <CCard className="mb-4">
              <CCardHeader className="d-flex align-items-center justify-content-between">
                <h5 className="mb-0">{user.fullName}</h5>
              </CCardHeader>
              <CCardBody>
                <div>User Id: {user.id}</div>
                <div>Email: {user.email}</div>
              </CCardBody>
            </CCard>
          </div>

          {/* CASES AND API KEYS COL */}
          <div className="col col-md-8">
            {/* CASES CARD */}
            <CCard className="mb-4">
              <CCardHeader className="d-flex align-items-center justify-content-between">
                <h5 className="mb-0">Cases</h5>
                <div>
                  <button
                    onClick={handleCreateCaseButtonClick}
                    className="btn btn-info me-1 text-white"
                    type="button"
                  >
                    <span className="btn-icon ">
                      <i className="cil-plus"></i>
                    </span>
                    New
                  </button>
                </div>
              </CCardHeader>
              <CCardBody>
                <CTable align="middle" className="mb-0 " hover responsive>
                  <CTableHead color="light">
                    <CTableRow>
                      <CTableHeaderCell>Case Id</CTableHeaderCell>
                      <CTableHeaderCell></CTableHeaderCell>
                    </CTableRow>
                  </CTableHead>
                  <CTableBody>
                    {cases.map((caseItem, index) => (
                      <CTableRow className="user-table-row" v-for="item in caseItems" key={index}>
                        <CTableDataCell>
                          <div>{caseItem.id}</div>
                        </CTableDataCell>
                        <CTableDataCell className="text-end">
                          <div>
                            <a
                              href={`/cases/${caseItem.id}`}
                              className="btn btn-primary"
                              type="button"
                            >
                              View
                            </a>
                            <button
                              onClick={() => handleDeleteCaseClick(caseItem.id)}
                              className="btn text-white btn-danger"
                              type="button"
                            >
                              Delete
                            </button>
                          </div>
                        </CTableDataCell>
                      </CTableRow>
                    ))}
                  </CTableBody>
                </CTable>
              </CCardBody>
            </CCard>

            {/* API KEYS CARD */}
            <CCard className="mb-4">
              <CCardHeader className="d-flex align-items-center justify-content-between">
                <h5 className="mb-0">API Keys</h5>
                <div>
                  <button
                    onClick={handleApiKeyButtonClick}
                    className="btn btn-info me-1 text-white"
                    type="button"
                  >
                    <span className="btn-icon ">
                      <i className="cil-plus"></i>
                    </span>
                    New
                  </button>
                </div>
              </CCardHeader>
              <CCardBody>
                <CTable align="middle" className="mb-0 " hover responsive>
                  <CTableHead color="light">
                    <CTableRow>
                      <CTableHeaderCell>Key</CTableHeaderCell>
                      <CTableHeaderCell></CTableHeaderCell>
                    </CTableRow>
                  </CTableHead>
                  <CTableBody>
                    {apiKeys.map((apiKey, index) => (
                      <CTableRow className="user-table-row" v-for="item in apiKeyItems" key={index}>
                        <CTableDataCell>
                          <div>{apiKey.key.slice(0, 40)} ...</div>
                        </CTableDataCell>
                        <CTableDataCell className="text-end">
                          <div>
                            <button
                              onClick={() => handleApiKeyDeleteClick(apiKey.id)}
                              className="btn text-white btn-danger"
                              type="button"
                            >
                              Delete
                            </button>
                          </div>
                        </CTableDataCell>
                      </CTableRow>
                    ))}
                  </CTableBody>
                </CTable>
              </CCardBody>
            </CCard>
          </div>
        </div>
      </div>
    </>
  )
}

export default User
