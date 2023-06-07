import React from 'react'
import axios from 'axios'
import {
  CButton,
  CCard,
  CCardBody,
  CCol,
  CContainer,
  CForm,
  CFormInput,
  CInputGroup,
  CInputGroupText,
  CRow,
} from '@coreui/react'
import CIcon from '@coreui/icons-react'
import { cilLockLocked, cilUser } from '@coreui/icons'
import { useNavigate } from 'react-router-dom'

import { API_URL } from 'src/const'

const defaultState = {
  fullName: '',
  email: '',
  password: '',
  repeatPassword: '',
}

const Register = () => {
  const [formData, setFormData] = React.useState(defaultState)
  const navigate = useNavigate()

  const handleInputChange = (e) => {
    setFormData((oldData) => ({
      ...oldData,
      [e.target.name]: e.target.value,
    }))
  }

  const handleCreateAccountClick = async () => {
    const data = formData
    delete data.repeatPassword
    const url = `${API_URL}/user`

    try {
      const res = await axios.post(url, data)
      if (res.status === 200) {
        navigate('/')
      } else {
        console.log(res)
      }
    } catch (e) {
      console.log(e)
    }
    console.log(formData)
  }

  return (
    <div className="bg-light min-vh-100 d-flex flex-row align-items-center">
      <CContainer>
        <CRow className="justify-content-center">
          <CCol md={9} lg={7} xl={6}>
            <CCard className="mx-4">
              <CCardBody className="p-4">
                <CForm>
                  <h1>Register</h1>
                  <p className="text-medium-emphasis">Create your account</p>
                  <CInputGroup className="mb-3">
                    <CInputGroupText>
                      <CIcon icon={cilUser} />
                    </CInputGroupText>
                    <CFormInput
                      onChange={handleInputChange}
                      value={formData.fullName}
                      placeholder="John Doe"
                      autoComplete="username"
                      name="fullName"
                    />
                  </CInputGroup>
                  <CInputGroup className="mb-3">
                    <CInputGroupText>@</CInputGroupText>
                    <CFormInput
                      onChange={handleInputChange}
                      value={formData.email}
                      name="email"
                      placeholder="Email"
                      autoComplete="email"
                    />
                  </CInputGroup>
                  <CInputGroup className="mb-3">
                    <CInputGroupText>
                      <CIcon icon={cilLockLocked} />
                    </CInputGroupText>
                    <CFormInput
                      value={formData.password}
                      name="password"
                      onChange={handleInputChange}
                      type="password"
                      placeholder="Password"
                      autoComplete="new-password"
                    />
                  </CInputGroup>
                  <CInputGroup className="mb-4">
                    <CInputGroupText>
                      <CIcon icon={cilLockLocked} />
                    </CInputGroupText>
                    <CFormInput
                      value={formData.repeatPassword}
                      name="repeatPassword"
                      onChange={handleInputChange}
                      type="password"
                      placeholder="Repeat password"
                      autoComplete="new-password"
                    />
                  </CInputGroup>
                  <div className="d-grid">
                    <CButton
                      className="text-white"
                      onClick={handleCreateAccountClick}
                      color="success"
                    >
                      Create Account
                    </CButton>
                  </div>
                </CForm>
              </CCardBody>
            </CCard>
          </CCol>
        </CRow>
      </CContainer>
    </div>
  )
}

export default Register
